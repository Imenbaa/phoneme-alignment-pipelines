import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import editdistance
from safetensors.torch import load_file

from transformers import Wav2Vec2ForPreTraining
from dataclasses import replace
os.environ["HF_HOME"] = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"] = "/vol/experiments/cache_imbenamor/datasets"
os.environ["TMPDIR"] = "/vol/experiments/cache_imbenamor/tmp"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import EarlyStoppingCallback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")  # no display needed on server
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import logging

# =========================
# Step 0: Config
# =========================
print("Step 0: Config")
exp_name = "w2vCTC_joint_nofxfy"
model_name    = "results/w2vCTC/checkpoint-23430"
output_dir    = f"results/{exp_name}"
vocab_name    = "vocab_w2vCTC.json"
preprocessed_path = "results/w2vCTC_FS/preprocessed"
RESUME_STEP = 12499
os.makedirs(output_dir, exist_ok=True)

# =========================
# Steps 1–6: Preprocessing (cached)
# =========================

if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset from disk...")
    dataset = load_from_disk(preprocessed_path)

else:
    print("Preprocessing from scratch...")

    print("Step 1: Loading dataset")
    dataset = load_dataset(
        "audiofolder",
        data_dir="/vol/experiments2/cbrazier/transcription/datasets/ester1_ester2_epac"
    )
    dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print("Step 2: Filtering UNK and very short seq")

    def is_valid(example):
        has_unk    = "|ø|n|k|" in example["phonemes"]
        too_short  = len(example["audio"]["array"]) < 24000
        few_phones = len(example["phoneme_single_v1"].split()) < 5
        return not has_unk and not too_short and not few_phones

    for split in ["train", "validation", "test"]:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(is_valid, num_proc=6)
        after  = len(dataset[split])
        print(f"  {split}: {before} → {after} ({before - after} removed)")

    print("Step 4: Build vocab")
    vocab_set = set()
    for split in ["train", "validation", "test"]:
        for seq in dataset[split]["phoneme_single_v1"]:
            vocab_set.update(seq.split())

    vocab_list = sorted(vocab_set)
    vocab_dict = {"[PAD]": 0, "[UNK]": 1}
    vocab_dict.update({v: i + 2 for i, v in enumerate(vocab_list)})

    with open(vocab_name, "w") as f:
        json.dump(vocab_dict, f)
    print(f"Vocab size: {len(vocab_dict)}")

    print("Step 5: Create processor")
    _tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_name,
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="",
    )
    _feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    _processor = Wav2Vec2Processor(
        feature_extractor=_feature_extractor, tokenizer=_tokenizer
    )

    print("Step 6: Preprocessing")

    def preprocess_batch(batch):
        audio_arrays   = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == 16000 for sr in sampling_rates)

        inputs = _feature_extractor(
            audio_arrays, sampling_rate=16000,
            return_tensors="np", padding=False, return_attention_mask=True,
        )
        with _processor.as_target_processor():
            labels = _processor(batch["phoneme_single_v1"]).input_ids

        return {
            "input_values":   inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels":         labels,
            "length":         [len(x) for x in inputs.input_values],
        }

    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].map(
            preprocess_batch,
            batched=True, batch_size=32, num_proc=6,
            remove_columns=dataset[split].column_names,
            cache_file_name=f"/vol/experiments/cache_imbenamor/map_FS_{split}.arrow",
        )

    print("Saving preprocessed dataset...")
    dataset.save_to_disk(preprocessed_path)

# Curriculum training — shortest utterances first
lengths        = dataset["train"]["length"]
sorted_indices = np.argsort(lengths)
sorted_dataset = dataset["train"].select(sorted_indices)

print(f"Min length: {min(lengths)/16000:.1f}s")
print(f"Max length: {max(lengths)/16000:.1f}s")

# =========================
# Always load processor
# =========================
print("Loading processor...")
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_name,
    unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000,
    padding_value=0.0, do_normalize=True, return_attention_mask=True,
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
)

# =========================
# Step 7: Data collator
# =========================
print("Step 7: Data collator")

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}          for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_values"] != 0).long()

        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# =========================
# Step 8: Metrics (PER)
# =========================
print("Step 8: Metrics")

def ctc_collapse(token_list, pad_token="[PAD]"):
    collapsed, prev = [], None
    for t in token_list:
        if t != prev and t != pad_token:
            collapsed.append(t)
        prev = t
    return collapsed

def compute_metrics(pred):
    pred_ids  = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids

    total_errors  = 0
    total_ref_len = 0

    for pred_id_seq, label_id_seq in zip(pred_ids, label_ids):
        pred_tokens = ctc_collapse(
            [processor.tokenizer.convert_ids_to_tokens(int(i)) for i in pred_id_seq]
        )
        label_tokens = [
            processor.tokenizer.convert_ids_to_tokens(int(i))
            for i in label_id_seq
            if i not in [-100, processor.tokenizer.pad_token_id]
        ]

        total_errors  += editdistance.eval(pred_tokens, label_tokens)
        total_ref_len += len(label_tokens)

    corpus_per = total_errors / max(total_ref_len, 1)
    return {"PER": float(corpus_per)}


# =========================
# Loss weights
# =========================
def get_loss_weights(global_step: int,
                     warmup_fs: int = 800,
                     warmup_m:  int = 5000):
    w_fs = min(1.0, global_step / max(warmup_fs, 1))
    w_m  = min(1.0, global_step / max(warmup_m,  1))
    return 5.0, 0.15 * w_fs, 0.5 * w_m  # λ_fs: 1.0 → 0.15


def initialize_quantizer(model, dataloader, device, n_batches=200):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            input_values   = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model.wav2vec2(input_values, attention_mask=attention_mask)
            X = outputs.last_hidden_state
            B, T, D = X.shape
            idx    = torch.randint(0, T, (B,))
            frames = X[torch.arange(B), idx]
            embeddings.append(frames.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    num_vars   = model.quantizer.num_vars
    perm       = torch.randperm(embeddings.shape[0])[:num_vars]

    model.quantizer.codevectors.copy_(embeddings[perm].to(device))
    model.quantizer.ema_embed.copy_(embeddings[perm].to(device))
    model.quantizer.cluster_size.fill_(1.0)

    print(f"EMA quantizer initialized from {embeddings.shape[0]} real frames")
    model.train()


class GumbelQuantizerEMA(nn.Module):
    def __init__(self, input_dim: int, num_vars: int = 320,
                 temp: float = 2.0, decay: float = 0.99):
        super().__init__()
        self.num_vars = num_vars
        self.temp     = temp
        self.decay    = decay

        self.weight_proj = nn.Linear(input_dim, num_vars)

        self.register_buffer("codevectors", torch.randn(num_vars, input_dim))
        self.register_buffer("cluster_size", torch.ones(num_vars))
        self.register_buffer("ema_embed",   torch.randn(num_vars, input_dim))
        nn.init.uniform_(self.codevectors, -1.0, 1.0)

    def forward(self, X: torch.Tensor):
        B, T, D = X.shape
        logits   = self.weight_proj(X)

        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temp, hard=True)

            with torch.no_grad():
                flat_probs = probs.reshape(-1, self.num_vars)
                flat_X     = X.reshape(-1, D)
                counts     = flat_probs.sum(0)
                embed_sum  = flat_probs.T @ flat_X

                self.cluster_size = (self.decay * self.cluster_size
                                     + (1 - self.decay) * counts)
                self.ema_embed    = (self.decay * self.ema_embed
                                     + (1 - self.decay) * embed_sum)
                self.codevectors  = (self.ema_embed
                                     / self.cluster_size.unsqueeze(1).clamp(min=1e-5))
        else:
            indices = logits.argmax(dim=-1)
            probs   = F.one_hot(indices, self.num_vars).float()

        quantized = torch.matmul(probs, self.codevectors)
        return quantized, probs


# =========================
# Step 9: Model
# =========================

class Wav2Vec2ForCTC_FS_REC(nn.Module):
    """
    Wav2Vec2 + Forward-Sum alignment + Gumbel-contrastive reconstruction.

    KEY CHANGES vs v1:
    ------------------
    1. phoneme_embed, fx, fy REMOVED.
       Y_emb is now computed by pooling X with CTC posteriors P(phoneme|frame).
       Since X and Y_emb are in the same space, no projection is needed.

    2. align_scale: learnable scalar temperature on the dot-product.
       Replaces the role of fx/fy with a single parameter.

    3. Diagonal prior added to alignment.
       Biases A toward the main diagonal from step 1, preventing attractor
       phonemes from absorbing frames across the whole sequence.

    4. Loss weights rebalanced:
       λ_ctc=3 (was 1) — stronger protection of the pretrained encoder.
       λ_fs=1  (was 0.5) — stronger monotonicity enforcement.

    Loss = λ_ctc * L_ctc + λ_fs * L_fs + λ_m * L_m
    """

    def __init__(
            self,
            base_model: Wav2Vec2ForCTC,
            vocab_size: int,
            hidden_dim: int,
            n_negatives: int = 50,
            temperature: float = 0.1,
    ):
        super().__init__()

        self.wav2vec2    = base_model.wav2vec2
        self.ctc_head    = base_model.lm_head
        self.n_negatives = n_negatives
        self.temperature = temperature
        self._keys_to_ignore_on_save = None
        self.fx = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.fx.weight)  # initialize as identity — no distortion at start


        # ── REMOVED: self.phoneme_embed, self.fx, self.fy ───────────────
        # Y_emb is built from X itself via CTC pooling — no extra params.

        # Learnable scalar: controls how sharp the dot-product becomes.
        # Initialized to 1 — equivalent to no scaling at the start.
        self.align_temperature = 5.0  # fixed — not learnable

        self.quantizer          = GumbelQuantizerEMA(input_dim=hidden_dim, decay=0.99)
        self.reconstruction_head = nn.Linear(2 * hidden_dim, hidden_dim)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.wav2vec2.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.wav2vec2.gradient_checkpointing_disable()

    def _ctc_decode_batch(self, logits: torch.Tensor) -> torch.Tensor:
        pred_ids  = logits.argmax(dim=-1)
        sequences = []
        for seq in pred_ids:
            unique_mask = torch.cat([
                torch.tensor([True], device=seq.device),
                seq[1:] != seq[:-1]
            ])
            collapsed = seq[unique_mask]
            collapsed = collapsed[collapsed != 0]
            if collapsed.numel() == 0:
                collapsed = torch.tensor([1], device=seq.device)
            sequences.append(collapsed)
        return torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0
        )

    # ------------------------------------------------------------------
    # NEW: CTC-pooled phoneme embeddings
    # ------------------------------------------------------------------
    def compute_phoneme_embeddings_from_ctc(
            self,
            X: torch.Tensor,           # (B, T, D) — detached from encoder graph
            logits: torch.Tensor,      # (B, T, vocab) — detached
            labels_clean: torch.Tensor # (B, N) — decoded phoneme IDs
    ) -> torch.Tensor:
        """
        For each phoneme n with ID p_n, compute its embedding as a weighted
        average of all frames, where the weight of frame t is P(p_n | t)
        from the CTC softmax.

        This gives contextual embeddings in the same space as X:
          - No attractor problem (norm is bounded by X norms)
          - Repeated phonemes get different embeddings per occurrence
          - No extra parameters
          - Works from step 1 because the pretrained encoder already has
            sharp CTC posteriors (~95.6% accuracy)
        """
        B, T, D = X.shape
        # (B, T, vocab) — CTC posterior probabilities
        ctc_probs = logits.softmax(dim=-1)

        Y_emb_list = []
        for b in range(B):
            phone_embs = []
            for p_id in labels_clean[b]:
                p_id = p_id.item()
                if p_id == 0:   # padding token
                    phone_embs.append(torch.zeros(D, device=X.device))
                    continue
                # Weight each frame by P(this phoneme | frame)
                weights = ctc_probs[b, :, p_id]                        # (T,)
                weights = weights / weights.sum().clamp(min=1e-8)       # normalize
                emb     = (weights.unsqueeze(-1) * X[b]).sum(0)         # (D,)
                phone_embs.append(emb)
            Y_emb_list.append(torch.stack(phone_embs))

        return torch.stack(Y_emb_list)   # (B, N, D)

    # ------------------------------------------------------------------
    # Masking (unchanged)
    # ------------------------------------------------------------------
    def mask_features(self, X: torch.Tensor, p_low: float = 0.1, p_high: float = 0.5):
        B, T, _ = X.shape
        mask_prob = torch.FloatTensor(1).uniform_(p_low, p_high).item()
        mask      = torch.rand(B, T, device=X.device) < mask_prob
        X_masked  = X.clone()
        X_masked[mask] = 0.0
        return X_masked, mask

    # ------------------------------------------------------------------
    # NEW: Alignment with direct dot-product + diagonal prior
    # ------------------------------------------------------------------
    def compute_alignment(self, X, Y_emb):
        B, T, D = X.shape
        _, N, _ = Y_emb.shape
    
        # Project X through learnable fx — this is what L_fs will optimize
        X_proj  = self.fx(X)                          # (B, T, D)
    
        # L2 normalize both
        X_norm  = F.normalize(X_proj, dim=-1)         # (B, T, D)
        Y_norm  = F.normalize(Y_emb,  dim=-1)         # (B, N, D)
    
        D_mat = self.align_temperature * torch.matmul(
            Y_norm, X_norm.transpose(1, 2)
        )                                              # (B, N, T)
    
        n_idx = torch.arange(N, device=X.device).float() / max(N - 1, 1)
        t_idx = torch.arange(T, device=X.device).float() / max(T - 1, 1)
        prior = -10.0 * (n_idx.unsqueeze(1) - t_idx.unsqueeze(0)) ** 2
        D_mat = D_mat + prior.unsqueeze(0)
    
        return torch.softmax(D_mat, dim=1)

    # ------------------------------------------------------------------
    # Forward-sum loss (unchanged)
    # ------------------------------------------------------------------
    def forward_sum_loss(self, A, label_lengths, input_lengths):
        B, N, T = A.shape
        blank_prob    = torch.full((B, 1, T), 1e-8, device=A.device, dtype=torch.float32)
        A_with_blank  = torch.cat([blank_prob, A.float()], dim=1)
        log_probs = torch.log(A_with_blank.clamp(min=1e-8)).permute(2, 0, 1)

        labels = torch.zeros(B, N, dtype=torch.long, device=A.device)
        for b in range(B):
            n = label_lengths[b].item()
            labels[b, :n] = torch.arange(1, n + 1, device=A.device)

        loss = F.ctc_loss(
            log_probs, labels, input_lengths, label_lengths,
            blank=0, reduction="none", zero_infinity=True,
        )
        loss = (loss / label_lengths.float().clamp(min=1)).mean()
        return loss

    # ------------------------------------------------------------------
    # Contrastive reconstruction loss (unchanged)
    # ------------------------------------------------------------------
    def contrastive_loss(self, H, X_quantized, mask, temperature=None):
        B, T, D  = H.shape
        temp     = temperature if temperature is not None else self.temperature
        H_flat   = H.reshape(B * T, D)
        Q_flat   = X_quantized.reshape(B * T, D)
        mask_flat = mask.reshape(B * T)

        if mask_flat.sum() == 0:
            return H_flat.sum() * 0.0

        H_norm = F.normalize(H_flat + 1e-8, dim=-1)
        Q_norm = F.normalize(Q_flat + 1e-8, dim=-1)

        masked_H = H_norm[mask_flat]
        masked_Q = Q_norm[mask_flat]
        M        = masked_H.shape[0]

        n_neg   = min(self.n_negatives, B * T - 1)
        neg_idx = torch.randint(0, B * T, (M, n_neg), device=H.device)
        Q_neg   = Q_norm[neg_idx]

        sim_pos = (masked_H * masked_Q).sum(dim=-1) / temp
        sim_neg = torch.bmm(
            Q_neg, masked_H.unsqueeze(-1)
        ).squeeze(-1) / temp

        all_sims = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        loss     = -sim_pos + torch.logsumexp(all_sims, dim=1)
        return loss.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_values, attention_mask=None, labels=None, global_step=0):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        X       = outputs.last_hidden_state          # (B, T, D)
        logits  = self.ctc_head(X)                   # (B, T, vocab)

        # Greedy CTC decode — no gradients needed
        with torch.no_grad():
            labels_clean = self._ctc_decode_batch(logits)

        # ── NEW: CTC-pooled phoneme embeddings ───────────────────────────
        # X and logits are detached so gradients do NOT flow back through
        # the pooling operation into the encoder a second time.
        Y_emb = self.compute_phoneme_embeddings_from_ctc(
            X.detach(), logits.detach(), labels_clean
        )                                            # (B, N, D)
        #with torch.no_grad():
            #self.align_scale.clamp_(0.1, 10.0)
        # ── NEW: Alignment with diagonal prior ───────────────────────────
        A = self.compute_alignment(X, Y_emb)         # (B, N, T)

        if labels is None:
            return {"loss": None, "logits": logits, "alignment": A}

        # Input / label lengths
        if attention_mask is not None:
            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1)
            ).long().clamp(min=1, max=X.shape[1])
        else:
            input_lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.long, device=X.device
            )

        label_lengths      = (labels != -100).sum(dim=-1)
        input_lengths      = torch.maximum(input_lengths, label_lengths).clamp(max=X.shape[1])
        label_lengths_pred = (labels_clean != 0).sum(dim=-1)

        # ── L_ctc: anchors encoder to phoneme recognition ────────────────
        loss_ctc = F.ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            labels, input_lengths, label_lengths,
            blank=0, reduction="mean", zero_infinity=True,
        )

        # ── L_fs: monotonic alignment ─────────────────────────────────────
        loss_fs = self.forward_sum_loss(A, label_lengths_pred, input_lengths)

        # ── L_m: contrastive reconstruction ──────────────────────────────
        X_quantized, _ = self.quantizer(X.detach())
        X_masked, mask = self.mask_features(X)
        YA             = torch.matmul(A.transpose(1, 2), Y_emb)   # (B, T, D)
        H              = self.reconstruction_head(torch.cat([X_masked, YA], dim=-1))
        current_temp   = max(0.1, 1.0 - 0.9 * min(1.0, global_step / 10000))
        self.quantizer.temp = current_temp
        loss_m = self.contrastive_loss(H, X_quantized, mask, temperature=0.1)
        effective_step = global_step + RESUME_STEP
        # ── Total loss ────────────────────────────────────────────────────
        λ_ctc, λ_fs, λ_m = get_loss_weights(effective_step)
        loss = λ_ctc * loss_ctc + λ_fs * loss_fs + λ_m * loss_m

        return {
            "loss":     loss,
            "logits":   logits,
            "alignment": A,
            "loss_ctc": loss_ctc.detach(),
            "loss_fs":  loss_fs.detach(),
            "loss_m":   loss_m.detach(),
        }


# =========================
# Load base model
# =========================
base_model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model = Wav2Vec2ForCTC_FS_REC(
    base_model,
    vocab_size  = len(processor.tokenizer),
    hidden_dim  = base_model.config.hidden_size,
    n_negatives = 50,
    temperature = 0.1,
)
model.wav2vec2.feature_extractor._freeze_parameters()

# =========================
# Step 10: Trainer
# =========================
print("Step 10: Trainer")

training_args = TrainingArguments(
    output_dir                  = output_dir,
    group_by_length             = False,
    length_column_name          = "length",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 16,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    lr_scheduler_type    = "constant_with_warmup", 
    warmup_steps                = 200,
    weight_decay                = 0.005,
    learning_rate               = 1e-4,
    num_train_epochs            = 30,
    metric_for_best_model       = "alignment_sharpness",
    greater_is_better           = True,
    logging_steps               = 1000,
    gradient_checkpointing      = True,
    max_grad_norm               = 2.0,
    fp16                        = False,
    bf16                        = True,
    report_to                   = "tensorboard",
    logging_dir                 = f"{output_dir}/tensorboard",
    load_best_model_at_end      = True,
    dataloader_num_workers      = 8,
    dataloader_pin_memory       = True,
    dataloader_prefetch_factor  = 2,
    remove_unused_columns       = False,
)


class FSCTCTrainer(Trainer):
    """
    Custom trainer with:
      - Per-step loss component logging (L_ctc, L_fs, L_m)
      - Per-epoch averaged loss logging (epoch/loss_*)
      - Posterior sharpness monitoring (early warning for encoder drift)
      - Alignment sharpness at evaluation
      - Gradient norm monitoring per module
    """

    def __init__(self, *args, log_every: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_every   = log_every
        self._step_count  = 0
        # Accumulator for epoch-level averaging
        self._epoch_losses = {"ctc": [], "fs": [], "m": []}

    def _get_eval_sampler(self, eval_dataset):
        return None

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_values   = inputs["input_values"],
            attention_mask = inputs.get("attention_mask"),
            labels         = inputs["labels"],
            global_step    = self.state.global_step,
        )
        loss     = outputs["loss"]
        loss_ctc = outputs.get("loss_ctc")
        loss_fs  = outputs.get("loss_fs")
        loss_m   = outputs.get("loss_m")

        # Accumulate for epoch logging
        if loss_ctc is not None:
            self._epoch_losses["ctc"].append(loss_ctc.item())
            self._epoch_losses["fs"].append(loss_fs.item())
            self._epoch_losses["m"].append(loss_m.item())

        self._step_count += 1
        if self._step_count % self._log_every == 0 and loss_ctc is not None:
            self._log_loss_components(loss, loss_ctc, loss_fs, loss_m, inputs, outputs)
            self._log_input_lengths(inputs, model)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Epoch-level logging — smooth, comparable across runs
    # ------------------------------------------------------------------
    def on_epoch_end(self, args, state, control, **kwargs):
        if not self._epoch_losses["ctc"]:
            return

        mean_ctc = np.mean(self._epoch_losses["ctc"])
        mean_fs  = np.mean(self._epoch_losses["fs"])
        mean_m   = np.mean(self._epoch_losses["m"])

        msg = (
            f"[epoch {int(state.epoch)}] "
            f"L_ctc={mean_ctc:.4f}  "
            f"L_fs={mean_fs:.4f}  "
            f"L_m={mean_m:.4f}"
        )
        print(msg)
        self.log({
            "epoch/loss_ctc": mean_ctc,
            "epoch/loss_fs":  mean_fs,
            "epoch/loss_m":   mean_m,
        })

        # Reset for next epoch
        self._epoch_losses = {"ctc": [], "fs": [], "m": []}

    # ------------------------------------------------------------------
    # Step-level loss logging + posterior sharpness
    # ------------------------------------------------------------------
    def _log_loss_components(self, loss, loss_ctc, loss_fs, loss_m, inputs, outputs):
        λ_ctc, λ_fs, λ_m = get_loss_weights(self.state.global_step)

        # Posterior sharpness: mean of max P(phoneme|frame) across the batch.
        # If this drops below ~0.5 the encoder is drifting and Y_emb degrades.
        with torch.no_grad():
            logits         = outputs["logits"]
            ctc_probs      = logits.softmax(dim=-1)
            max_probs      = ctc_probs.max(dim=-1).values
            post_sharpness = max_probs.mean().item()

        msg = (
            f"[step {self._step_count}] "
            f"loss={loss.item():.4f}  "
            f"L_ctc={loss_ctc.item():.4f} (w={λ_ctc:.1f})  "
            f"L_fs={loss_fs.item():.4f} (w={λ_fs:.2f})  "
            f"L_m={loss_m.item():.4f} (w={λ_m:.2f})  "
            f"posterior_sharpness={post_sharpness:.3f}"
        )
        print(msg)
        # In _log_loss_components:
        #self.log({"train/align_scale": model.align_scale.item()})
        #print(f"  align_scale={model.align_scale.item():.4f}")
        self.log({
            "train/loss_ctc":            loss_ctc.item(),
            "train/loss_fs":             loss_fs.item(),
            "train/loss_m":              loss_m.item(),
            "train/weighted_ctc":        (λ_ctc * loss_ctc).item(),
            "train/weighted_fs":         (λ_fs  * loss_fs).item(),
            "train/weighted_m":          (λ_m   * loss_m).item(),
            "train/posterior_sharpness": post_sharpness,
        })

        # Sanity warnings
        if loss_ctc.item() < 0.01:
            print("  ⚠  L_ctc near zero — possible label/blank collapse")
        if loss_fs.item() > 50.0:
            print("  ⚠  L_fs very large — alignment not started learning")
        if post_sharpness < 0.5:
            print("  ⚠  Posterior sharpness low — encoder drifting, Y_emb degrading")
        if torch.isnan(loss):
            print("  ✗  NaN in total loss")

    # ------------------------------------------------------------------
    # Input length sanity check
    # ------------------------------------------------------------------
    def _log_input_lengths(self, inputs, model):
        with torch.no_grad():
            attn = inputs.get("attention_mask")
            if attn is None:
                return
            feat_len  = model.wav2vec2._get_feat_extract_output_lengths(
                attn.sum(dim=-1)
            ).float()
            label_len = (inputs["labels"] != -100).sum(dim=-1).float()

        violations = (feat_len < label_len).sum().item()
        if violations > 0:
            print(f"  ✗  {violations} samples where frame_len < label_len")
©
    # ------------------------------------------------------------------
    # Optimizer — 
    # ------------------------------------------------------------------
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {"params": list(self.model.wav2vec2.parameters()), "lr": 1e-7},
            {"params": list(self.model.ctc_head.parameters()),  "lr": 1e-7},
            {"params": list(self.model.fx.parameters()),        "lr": 1e-4},  # ← L_fs optimizes this
            {"params": list(self.model.reconstruction_head.parameters()), "lr": 1e-4},
            {"params": list(self.model.quantizer.weight_proj.parameters()), "lr": 1e-3},
        ], weight_decay=self.args.weight_decay)
        return self.optimizer

    # ------------------------------------------------------------------
    # training_step: gradient norm monitoring
    # ------------------------------------------------------------------
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        if self._step_count % self._log_every == 0:
            norms = {}
            for name, module in [
                ("encoder",    model.wav2vec2.encoder),
                ("ctc_head",   model.ctc_head),
                ("quantizer",  model.quantizer),
                ("recon_head", model.reconstruction_head),
            ]:
                #if name == "align_scale":
                    #g = model.align_scale.grad
                    #norms[name] = g.abs().item() if g is not None else 0.0
                    #continue
                total, count = 0.0, 0
                for p in module.parameters():
                    if p.grad is not None:
                        total += p.grad.norm(2).item() ** 2
                        count += 1
                norms[name] = total ** 0.5 if count > 0 else 0.0

            print("  grad norms — " + "  ".join(f"{k}: {v:.3f}" for k, v in norms.items()))

            dead = [k for k, v in norms.items() if v < 1e-8]
            if dead:
                print(f"  ⚠  zero gradients in: {dead}")

        return loss

    # ------------------------------------------------------------------
    # Alignment sharpness at evaluation
    # ------------------------------------------------------------------
    def evaluate(self, *args, **kwargs):
        sharpness = self._log_alignment_sharpness()
        metrics   = super().evaluate(*args, **kwargs)
        if sharpness is not None:
            metrics["eval_alignment_sharpness"] = sharpness
        return metrics

    def _log_alignment_sharpness(self):
        model     = self.model
        model.eval()
        sharpness = None
        try:
            dl    = self.get_eval_dataloader()
            batch = next(iter(dl))
            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                outputs      = model.wav2vec2(
                    batch["input_values"],
                    attention_mask=batch.get("attention_mask")
                )
                X            = outputs.last_hidden_state
                logits       = model.ctc_head(X)
                labels_clean = model._ctc_decode_batch(logits)
                Y_emb        = model.compute_phoneme_embeddings_from_ctc(
                    X.detach(), logits.detach(), labels_clean
                )
                A            = model.compute_alignment(X, Y_emb)
                self._plot_alignment_to_tensorboard(A, labels_clean, epoch=int(self.state.epoch))
                eps       = 1e-8
                entropy   = -(A * (A + eps).log()).sum(dim=1)
                mean_H    = entropy.mean().item()
                max_H     = np.log(labels_clean.shape[1] + eps)
                sharpness = 100 * (1 - mean_H / max_H)

            print(
                f"\n[alignment @ step {self._step_count}] "
                f"entropy: {mean_H:.3f}/{max_H:.3f}  "
                f"sharpness: {sharpness:.1f}%\n"
            )
            self.log({"alignment_sharpness": sharpness})

        except Exception as e:
            print(f"Alignment sharpness logging failed: {e}")
        finally:
            model.train()

        return sharpness

    # ------------------------------------------------------------------
    # prediction_step (unchanged)
    # ------------------------------------------------------------------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(
                input_values   = inputs["input_values"],
                attention_mask = inputs.get("attention_mask"),
                labels         = inputs["labels"],
            )
        loss   = outputs["loss"]
        logits = outputs["logits"]
        labels = inputs["labels"]

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)
    def _plot_alignment_to_tensorboard(self, A, labels_clean, epoch):
        """
        Plot one alignment matrix per epoch to TensorBoard.
        Picks the first sample in the eval batch.
        """
        writer = SummaryWriter(log_dir=self.args.logging_dir)
    
        # Take first sample only
        n_phones = (labels_clean[0] != 0).sum().item()
        A_sample = A[0, :n_phones, :].cpu().float().numpy()  # (N, T)
    
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            A_sample,
            aspect="auto",
            origin="lower",
            cmap="hot",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax)
    
        # Argmax path
        path = A[0, :n_phones, :].argmax(dim=0).cpu().numpy()  # (T,)
        ax.plot(range(len(path)), path, color="lime", linewidth=1.0, label="argmax path")
    
        # Ideal diagonal
        T = A_sample.shape[1]
        ax.plot(
            range(T),
            np.linspace(0, n_phones - 1, T),
            color="deepskyblue", linewidth=1.0, linestyle="--", label="ideal diagonal"
        )
    
        ax.set_xlabel("Frames (×20ms)")
        ax.set_ylabel("Phoneme position")
        ax.set_title(f"Alignment — epoch {epoch}  (N={n_phones} phones, T={T} frames)")
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
    
        writer.add_figure("alignment/matrix", fig, global_step=epoch)
        writer.close()
        plt.close(fig)


# =========================
# Launch
# =========================
trainer = FSCTCTrainer(
    model           = model,
    data_collator   = data_collator,
    args            = training_args,
    compute_metrics = compute_metrics,
    train_dataset   = sorted_dataset,
    eval_dataset    = dataset["validation"],
    log_every       = 1000,
    tokenizer       = processor.feature_extractor,
)

initialize_quantizer(
    model      = model,
    dataloader = trainer.get_train_dataloader(),
    device     = next(model.parameters()).device,
    n_batches  = 200,
)

checkpoint_dir = "results/w2vCTC_joint_nofxfy/checkpoint-12499"  # use latest

state_dict = load_file(f"{checkpoint_dir}/model.safetensors", device="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing:    {missing}")
print(f"Unexpected: {unexpected}")

trainer.train()

logs_path = os.path.join(output_dir, "logs.json")
with open(logs_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)
print(f"Training complete. Logs saved to {logs_path}")