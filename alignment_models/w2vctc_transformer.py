

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import editdistance

os.environ["HF_HOME"]            = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"]  = "/vol/experiments/cache_imbenamor/datasets"
os.environ["TMPDIR"]             = "/vol/experiments/cache_imbenamor/tmp"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# =============================================================================
# Step 0: Config
# =============================================================================
print("Step 0: Config")

RESUME_STEP = 0   # training from scratch

exp_name          = "w2vCTC_phonetransformer"
model_name        = "results/w2vCTC/checkpoint-23430"
output_dir        = f"results/{exp_name}"
vocab_name        = "vocab_w2vCTC.json"
preprocessed_path = "results/w2vCTC_FS/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# NOTE: YOUR_TO_IPA and XPhoneBERT removed — PhonemeTransformer operates
# directly on CTC token IDs with no IPA conversion needed.


# =============================================================================
# Dataset (cached — identical to bertphone script)
# =============================================================================
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

    print("Step 2: Filtering")
    def is_valid(example):
        has_unk    = "|ø|n|k|" in example["phonemes"]
        too_short  = len(example["audio"]["array"]) < 24000
        few_phones = len(example["phoneme_single_v1"].split()) < 5
        return not has_unk and not too_short and not few_phones

    for split in ["train", "validation", "test"]:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(is_valid, num_proc=6)
        print(f"  {split}: {before} → {len(dataset[split])}")

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

lengths        = dataset["train"]["length"]
sorted_indices = np.argsort(lengths)
sorted_dataset = dataset["train"].select(sorted_indices)
print(f"Min length: {min(lengths)/16000:.1f}s")
print(f"Max length: {max(lengths)/16000:.1f}s")


# =============================================================================
# Processor
# =============================================================================
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


# =============================================================================
# Step 7: Data collator
# =============================================================================
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


# =============================================================================
# Step 8: Metrics (PER)
# =============================================================================
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
    return {"PER": float(total_errors / max(total_ref_len, 1))}


# =============================================================================
# Loss weights
# lambda_fs raised to 1.0 (was 0.15) — now the primary alignment driver
# =============================================================================
def get_loss_weights(global_step: int, warmup_fs: int = 800, warmup_m: int = 5000):
    w_m  = min(1.0, global_step / max(warmup_m,  1))
    return 5.0, 1.0 , 0.5 * w_m   # lambda_fs = 1.0 (was 0.15)


# =============================================================================
# Standalone loss functions
# (computed in training loop, not inside forward())
# =============================================================================

def forward_sum_loss(A: torch.Tensor,
                     labels_clean: torch.Tensor) -> torch.Tensor:
    """
    Monotonic forward-sum loss on alignment matrix A.
    A             : (B, N, T) — softmax over phonemes (dim=1)
    labels_clean  : (B, N_pad) — 0 = PAD
    Returns scalar (mean over batch).
    """
    B, N, T = A.shape
    log_A   = torch.log(A.clamp(min=1e-8))   # (B, N, T)
    losses  = []

    for b in range(B):
        n_valid = int((labels_clean[b] != 0).sum().item())
        if n_valid < 2:
            continue
        la = log_A[b, :n_valid, :]   # (n, T)

        # Forward algorithm in log-space (vectorised over n)
        NEG_INF = torch.full((1,), -1e9, device=A.device, dtype=A.dtype)
        alpha   = torch.full((n_valid,), -1e9, device=A.device, dtype=A.dtype)
        alpha[0] = la[0, 0]

        for t in range(1, T):
            advance = torch.cat([NEG_INF, alpha[:-1]])
            alpha   = torch.logaddexp(alpha, advance) + la[:, t]

        log_prob = alpha[n_valid - 1]
        losses.append(-log_prob / (n_valid + T))

    if not losses:
        return (A * 0).sum()   # zero, keeps grad graph intact
    return torch.stack(losses).mean()





def reconstruction_loss(model, X: torch.Tensor, A: torch.Tensor,
                        Y_emb: torch.Tensor,
                        mask_prob: float = 0.1) -> torch.Tensor:
    """
    Contrastive reconstruction loss at masked frames.
    Predicts quantized acoustic target q_t from [X_masked; Y_A].
    X     : (B, T, D)
    A     : (B, N, T)
    Y_emb : (B, N, D)
    """
    B, T, D = X.shape

    # Random frame mask
    mask = torch.rand(B, T, device=X.device) < mask_prob
    if not mask.any():
        return (X * 0).sum()

    # Quantized targets (stop-grad)
    with torch.no_grad():
        q, _ = model.quantizer(X)    # (B, T, D)

    # Phoneme-aligned representation: Y_A[b, t] = sum_n A[b,n,t] * Y_emb[b,n]
    Y_A = torch.bmm(A.transpose(1, 2), Y_emb)   # (B, T, D)

    # Mask X
    X_masked = X.clone()
    X_masked[mask] = 0.0

    # Reconstruct
    combined = torch.cat([X_masked, Y_A], dim=-1)   # (B, T, 2D)
    H        = model.reconstruction_head(combined)   # (B, T, D)

    H_masked = F.normalize(H[mask], dim=-1)    # (M, D)
    q_pos    = F.normalize(q[mask], dim=-1)    # (M, D)
    M        = H_masked.shape[0]

    n_neg = min(model.n_negatives, M - 1)
    if n_neg <= 0:
        return (X * 0).sum()

    # Negative sampling
    neg_idx    = torch.randint(0, M, (M, n_neg), device=X.device)
    q_neg      = q_pos[neg_idx]                              # (M, n_neg, D)
    pos_scores = (H_masked * q_pos).sum(-1, keepdim=True)   # (M, 1)
    neg_scores = torch.bmm(q_neg, H_masked.unsqueeze(-1)).squeeze(-1)  # (M, n_neg)
    all_scores = torch.cat([pos_scores, neg_scores], dim=1) / model.temperature
    targets    = torch.zeros(M, dtype=torch.long, device=X.device)

    return F.cross_entropy(all_scores, targets)


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
            B, T, _ = X.shape
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


# =============================================================================
# Step 9: Model
# =============================================================================

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
        return torch.matmul(probs, self.codevectors), probs


class PhonemeTransformer(nn.Module):
    """
    Lightweight fully-trainable phoneme encoder.
    Replaces XPhoneBERT + bert_proj entirely.
    Input : CTC token IDs (vocab indices, no IPA conversion needed).
    Output: contextual phoneme embeddings in audio encoder space.
    """
    def __init__(self, vocab_size: int, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 3,
                 out_dim: int = 1024, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)   # learned positions
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,    # Pre-LN: more stable training from scratch
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.out_proj    = nn.Linear(d_model, out_dim)
        self._init_weights()

    def _init_weights(self):
        # Small init: avoids cosine similarity saturation at step 0
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, token_ids: torch.Tensor,
                key_padding_mask: torch.Tensor = None):
        """
        token_ids       : (B, N)  — CTC token ids, 0 = PAD
        key_padding_mask: (B, N)  — True where PAD
        Returns         : (B, N, out_dim)
        """
        B, N = token_ids.shape
        pos  = torch.arange(N, device=token_ids.device).unsqueeze(0)
        x    = self.embedding(token_ids) + self.pos_embed(pos)
        x    = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.out_proj(x)


class Wav2Vec2ForCTC_FS_Transformer(nn.Module):
    def __init__(self, base_model, vocab_size: int, hidden_dim: int,
                 ctc_tokenizer,
                 ph_d_model: int = 256, ph_layers: int = 3,
                 n_negatives: int = 50, temperature: float = 0.1):
        super().__init__()

        # Audio encoder — pretrained, near-frozen
        self.wav2vec2      = base_model.wav2vec2
        self.ctc_head      = base_model.lm_head
        self.ctc_tokenizer = ctc_tokenizer
        self.n_negatives   = n_negatives
        self.temperature   = temperature

        # Alignment hyper-parameters — must match inference
        self.align_temperature = 15.0
        self.gaussian_weight   = 30.0
        self.ahead_weight      = 40.0

        # Trainable phoneme encoder (replaces XPhoneBERT + bert_proj)
        self.phoneme_encoder = PhonemeTransformer(
            vocab_size = vocab_size,
            d_model    = ph_d_model,
            nhead      = 4,
            num_layers = ph_layers,
            out_dim    = hidden_dim,
        )

    
        # Trainable audio projection — identity init so alignment starts neutral
        self.fx = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.fx.weight)

        # Contrastive reconstruction head
        self.quantizer           = GumbelQuantizerEMA(input_dim=hidden_dim)
        self.reconstruction_head = nn.Linear(2 * hidden_dim, hidden_dim)
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.wav2vec2.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        self.wav2vec2.gradient_checkpointing_disable()
    # ------------------------------------------------------------------
    def _ctc_decode_batch(self, logits: torch.Tensor) -> torch.Tensor:
        pred_ids  = logits.argmax(dim=-1)
        sequences = []
        for seq in pred_ids:
            unique_mask = torch.cat([
                torch.tensor([True], device=seq.device),
                seq[1:] != seq[:-1],
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
    def compute_phoneme_embeddings(self, labels_clean: torch.Tensor) -> torch.Tensor:
        """
        labels_clean : (B, N_pad) — CTC token ids, 0 = PAD
        Returns      : (B, N_pad, hidden_dim)
        """
        pad_mask = (labels_clean == 0)
        return self.phoneme_encoder(labels_clean, key_padding_mask=pad_mask)

    # ------------------------------------------------------------------
    def compute_alignment(self, X: torch.Tensor, Y_emb: torch.Tensor,
                          labels_clean: torch.Tensor) -> torch.Tensor:
        """
        Exact training formula:
          D = 15 * cosine(Y_emb, fx(X))
          prior = gaussian(-30) + ahead_penalty(-40)
          A = softmax(D + prior, dim=1)
        PAD rows masked to -inf so they never win softmax.
        """
        B, T, D = X.shape
        _, N, _ = Y_emb.shape

        X_proj = self.fx(X)
        X_norm = F.normalize(X_proj, dim=-1)
        Y_norm = F.normalize(Y_emb,  dim=-1)
        D_mat  = self.align_temperature * torch.matmul(
            Y_norm, X_norm.transpose(1, 2)
        )   # (B, N, T)

        # Mask PAD phoneme rows
        pad_mask = (labels_clean == 0).unsqueeze(-1)   # (B, N, 1)
        D_mat    = D_mat.masked_fill(pad_mask, float("-inf"))

        n_idx = torch.arange(N, device=X.device).float() / max(N - 1, 1)
        t_idx = torch.arange(T, device=X.device).float() / max(T - 1, 1)
        diff  = n_idx.unsqueeze(1) - t_idx.unsqueeze(0)   # (N, T)

        prior = (-self.gaussian_weight * diff ** 2
                 - self.ahead_weight   * torch.clamp(diff, min=0) ** 2)

        return torch.softmax(D_mat + prior.unsqueeze(0), dim=1)   # (B, N, T)

    # ------------------------------------------------------------------
    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        Returns (logits, A, labels_clean, Y_emb, X).
        All losses are computed externally in compute_loss().
        """
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        X       = outputs.last_hidden_state        # (B, T, 1024)
        logits  = self.ctc_head(X)

        with torch.no_grad():
            labels_clean = self._ctc_decode_batch(logits)

        Y_emb = self.compute_phoneme_embeddings(labels_clean)
        A     = self.compute_alignment(X, Y_emb, labels_clean)
        return logits, A, labels_clean, Y_emb, X


# =============================================================================
# Load base model (pretrained CTC encoder — no joint checkpoint)
# =============================================================================
base_model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction = "mean",
    ctc_zero_infinity  = True,
    pad_token_id       = processor.tokenizer.pad_token_id,
    vocab_size         = len(processor.tokenizer),
)

model = Wav2Vec2ForCTC_FS_Transformer(
    base_model    = base_model,
    vocab_size    = len(processor.tokenizer),
    hidden_dim    = base_model.config.hidden_size,
    ctc_tokenizer = processor.tokenizer,
    ph_d_model    = 256,
    ph_layers     = 3,
    n_negatives   = 50,
    temperature   = 0.1,
)
model.wav2vec2.feature_extractor._freeze_parameters()

# No checkpoint resume — training from scratch.
# PhonemeTransformer starts from random init (xavier / small normal).
# fx starts from identity.
# Quantizer will be initialized from real audio frames below.

print(f"Model parameters:")
for name, group_params in [
    ("wav2vec2",         model.wav2vec2.parameters()),
    ("ctc_head",         model.ctc_head.parameters()),
    ("phoneme_encoder",  model.phoneme_encoder.parameters()),
    ("fx",               model.fx.parameters()),
    ("reconstruction",   model.reconstruction_head.parameters()),
    ("quantizer",        model.quantizer.parameters()),
]:
    n = sum(p.numel() for p in group_params)
    print(f"  {name:<20}: {n:>10,}")


# =============================================================================
# Step 10: Trainer
# =============================================================================
print("Step 10: Trainer")

training_args = TrainingArguments(
    output_dir                  = output_dir,
    group_by_length             = False,
    length_column_name          = "length",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 64,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    lr_scheduler_type           = "constant_with_warmup",
    warmup_steps                = 500,          # re-enabled: training from scratch
    weight_decay                = 0.005,
    learning_rate               = 1e-4,
    num_train_epochs            = 30,
    metric_for_best_model       = "PER",
    greater_is_better           = False,
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

    def __init__(self, *args, log_every: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_every    = log_every
        self._step_count   = 0
        self._epoch_losses = {"ctc": [], "fs": [], "m": []}

    def _get_eval_sampler(self, eval_dataset):
        return None

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        input_values   = inputs["input_values"]
        attention_mask = inputs.get("attention_mask")
        labels         = inputs["labels"]   # (B, L) with -100 for padding

        # ── Forward pass ─────────────────────────────────────────────
        logits, A, labels_clean, Y_emb, X = model(input_values, attention_mask)

        # ── L_ctc ────────────────────────────────────────────────────
        input_lengths = model.wav2vec2._get_feat_extract_output_lengths(
            attention_mask.sum(-1) if attention_mask is not None
            else torch.full((logits.shape[0],), logits.shape[1],
                            device=logits.device)
        ).long()
        label_lengths = (labels != -100).sum(-1)
        labels_ctc    = labels.clone()
        labels_ctc[labels_ctc == -100] = 0

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
        l_ctc = F.ctc_loss(
            log_probs, labels_ctc, input_lengths, label_lengths,
            blank     = model.ctc_tokenizer.pad_token_id,
            reduction = "mean",
            zero_infinity = True,
        )

        # ── L_fs ─────────────────────────────────────────────────────
        l_fs = forward_sum_loss(A, labels_clean)

        # ── L_m ──────────────────────────────────────────────────────
        l_m = reconstruction_loss(model, X, A, Y_emb)

        # ── L_div (phoneme diversity) ─────────────────────────────────

        # ── Weighted sum ──────────────────────────────────────────────
        λ_ctc, λ_fs, λ_m = get_loss_weights(
            self.state.global_step + RESUME_STEP
        )
        loss = λ_ctc * l_ctc + λ_fs * l_fs + λ_m * l_m 

        # ── Logging ───────────────────────────────────────────────────
        self._epoch_losses["ctc"].append(l_ctc.item())
        self._epoch_losses["fs"].append(l_fs.item())
        self._epoch_losses["m"].append(l_m.item())
        self._step_count += 1

        if self._step_count % self._log_every == 0:
            self._log_loss_components(
                loss, l_ctc, l_fs, l_m, 
                λ_ctc, λ_fs, λ_m, logits, Y_emb, labels_clean
            )
            self._log_input_lengths(inputs, model)

        if return_outputs:
            return loss, {"logits": logits, "A": A,
                          "labels_clean": labels_clean}
        return loss

    # ------------------------------------------------------------------
    def on_epoch_end(self, args, state, control, **kwargs):
        if not self._epoch_losses["ctc"]:
            return
        means = {k: np.mean(v) for k, v in self._epoch_losses.items()}
        msg = (
            f"[epoch {int(state.epoch)}] "
            f"L_ctc={means['ctc']:.4f}  L_fs={means['fs']:.4f}  "
            f"L_m={means['m']:.4f} "
        )
        print(msg)
        self.log({f"epoch/loss_{k}": v for k, v in means.items()})
        self._epoch_losses = {"ctc": [], "fs": [], "m": []}

    # ------------------------------------------------------------------
    def _log_loss_components(self, loss, l_ctc, l_fs, l_m,
                              λ_ctc, λ_fs, λ_m, logits, Y_emb, labels_clean):
        # CTC posterior sharpness
        with torch.no_grad():
            ctc_probs      = logits.softmax(dim=-1)
            post_sharpness = ctc_probs.max(dim=-1).values.mean().item()

        # Phoneme embedding cosine similarity (diversity monitor)
        with torch.no_grad():
            b = 0
            n_valid = int((labels_clean[b] != 0).sum().item())
            if n_valid >= 2:
                Y_norm   = F.normalize(Y_emb[b, :n_valid], dim=-1)
                gram     = torch.matmul(Y_norm, Y_norm.T)
                mask     = ~torch.eye(n_valid, dtype=torch.bool,
                                      device=Y_emb.device)
                avg_sim  = gram[mask].mean().item()
            else:
                avg_sim  = float("nan")

        msg = (
            f"[step {self._step_count}] "
            f"loss={loss.item():.4f}  "
            f"L_ctc={l_ctc.item():.4f}(w={λ_ctc:.1f})  "
            f"L_fs={l_fs.item():.4f}(w={λ_fs:.2f})  "
            f"L_m={l_m.item():.4f}(w={λ_m:.2f})  "
            f"post_sharp={post_sharpness:.3f}  "
            f"avg_ph_cos={avg_sim:.3f}"   # target: < 0.3
        )
        print(msg)

        self.log({
            "train/loss_ctc":            l_ctc.item(),
            "train/loss_fs":             l_fs.item(),
            "train/loss_m":              l_m.item(),
            "train/weighted_ctc":        (λ_ctc * l_ctc).item(),
            "train/weighted_fs":         (λ_fs  * l_fs).item(),
            "train/weighted_m":          (λ_m   * l_m).item(),
            "train/posterior_sharpness": post_sharpness,
            "train/avg_phoneme_cos_sim": avg_sim,
        })

        # Warnings
        if l_ctc.item() < 0.01:
            print("  ⚠  L_ctc near zero")
        if post_sharpness < 0.5:
            print("  ⚠  Posterior sharpness low")
        if avg_sim > 0.5:
            print("  ⚠  avg_ph_cos > 0.5 — embeddings collapsing, raise lambda_div")
        if torch.isnan(loss):
            print("  ✗  NaN in total loss")

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

    # ------------------------------------------------------------------
    def create_optimizer(self):
        # BUG FIX: original returned self.optimizer instead of optimizer
        optimizer = torch.optim.AdamW([
            {"params": model.wav2vec2.parameters(),              "lr": 1e-7},
            {"params": model.ctc_head.parameters(),              "lr": 1e-7},
            {"params": model.fx.parameters(),                    "lr": 1e-4},
            {"params": model.phoneme_encoder.parameters(),       "lr": 2e-3},
            {"params": model.reconstruction_head.parameters(),   "lr": 1e-4},
            {"params": model.quantizer.weight_proj.parameters(), "lr": 1e-3},
        ], weight_decay=0.01)
        self.optimizer = optimizer
        return optimizer

    # ------------------------------------------------------------------
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self._step_count % self._log_every == 0:
            norms = {}
            for name, module in [
                ("encoder",        model.wav2vec2.encoder),
                ("ctc_head",       model.ctc_head),
                ("phoneme_enc",    model.phoneme_encoder),   # NEW
                ("quantizer",      model.quantizer),
                ("recon_head",     model.reconstruction_head),
            ]:
                total, count = 0.0, 0
                for p in module.parameters():
                    if p.grad is not None:
                        total += p.grad.norm(2).item() ** 2
                        count += 1
                norms[name] = total ** 0.5 if count > 0 else 0.0

            print("  grad norms — "
                  + "  ".join(f"{k}: {v:.3f}" for k, v in norms.items()))
            dead = [k for k, v in norms.items() if v < 1e-8]
            if dead:
                print(f"  ⚠  zero gradients in: {dead}")
        return loss

    # ------------------------------------------------------------------
    def evaluate(self, *args, **kwargs):
        sharpness = self._log_alignment_sharpness()
        metrics   = super().evaluate(*args, **kwargs)
        if sharpness is not None:
            metrics["eval_alignment_sharpness"] = sharpness
        return metrics

    # ------------------------------------------------------------------
    def _log_alignment_sharpness(self):
        model     = self.model
        model.eval()
        sharpness = None
        try:
            dl    = self.get_eval_dataloader()
            batch = next(iter(dl))
            batch = self._prepare_inputs(batch)
            with torch.no_grad():
                # New forward signature: returns 5-tuple
                logits, A, labels_clean, Y_emb, X = model(
                    batch["input_values"],
                    attention_mask=batch.get("attention_mask"),
                )
                self._plot_alignment_to_tensorboard(
                    A, labels_clean, epoch=int(self.state.epoch)
                )
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
    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]

        with torch.no_grad():
            logits, A, labels_clean, Y_emb, X = model(
                input_values   = inputs["input_values"],
                attention_mask = inputs.get("attention_mask"),
            )
            # Compute CTC loss for eval reporting
            attention_mask = inputs.get("attention_mask")
            input_lengths  = model.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(-1) if attention_mask is not None
                else torch.full((logits.shape[0],), logits.shape[1],
                                device=logits.device)
            ).long()
            label_lengths = (labels != -100).sum(-1)
            labels_ctc    = labels.clone()
            labels_ctc[labels_ctc == -100] = 0
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = F.ctc_loss(
                log_probs, labels_ctc, input_lengths, label_lengths,
                blank         = model.ctc_tokenizer.pad_token_id,
                reduction     = "mean",
                zero_infinity = True,
            )

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    # ------------------------------------------------------------------
    def _plot_alignment_to_tensorboard(self, A, labels_clean, epoch):
        writer   = SummaryWriter(log_dir=self.args.logging_dir)
        n_phones = int((labels_clean[0] != 0).sum().item())
        A_sample = A[0, :n_phones, :].cpu().float().numpy()
        fig, ax  = plt.subplots(figsize=(12, 4))
        im       = ax.imshow(A_sample, aspect="auto", origin="lower",
                             cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        path = A[0, :n_phones, :].argmax(dim=0).cpu().numpy()
        T    = A_sample.shape[1]
        ax.plot(range(len(path)), path,
                color="lime", linewidth=1.0, label="argmax path")
        ax.plot(range(T), np.linspace(0, n_phones - 1, T),
                color="deepskyblue", linewidth=1.0, linestyle="--",
                label="ideal diagonal")
        ax.set_xlabel("Frames (×20ms)")
        ax.set_ylabel("Phoneme position")
        ax.set_title(
            f"Alignment — epoch {epoch}  "
            f"(N={n_phones} phones, T={T} frames)"
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        writer.add_figure("alignment/matrix", fig, global_step=epoch)
        writer.close()
        plt.close(fig)


# =============================================================================
# Launch
# =============================================================================
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

# Initialize quantizer from real audio frames before training starts
print("Initializing quantizer from real data...")
train_dl = trainer.get_train_dataloader()
initialize_quantizer(model, train_dl, device="cuda", n_batches=200)

trainer.train()

logs_path = os.path.join(output_dir, "logs.json")
with open(logs_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)
print(f"Training complete. Logs saved to {logs_path}")