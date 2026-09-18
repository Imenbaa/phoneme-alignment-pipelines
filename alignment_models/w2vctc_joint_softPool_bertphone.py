import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import editdistance
from transformers import Wav2Vec2ForCTC, AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Union
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)

os.environ["HF_HOME"]             = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"]   = "/vol/experiments/cache_imbenamor/datasets"
os.environ["TMPDIR"]              = "/vol/experiments/cache_imbenamor/tmp"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =========================
# Step 0: Config
# =========================
print("Step 0: Config")
RESUME_STEP = 0   # set to checkpoint step when resuming

exp_name          = "w2vCTC_softpool_xphonebert"
model_name        = "results/w2vCTC/checkpoint-23430"
output_dir        = f"results/{exp_name}"
vocab_name        = "vocab_w2vCTC-woSIL.json"
preprocessed_path = "results/w2vCTC_FS/preprocessed"

os.makedirs(output_dir, exist_ok=True)

# IPA mapping: corpus vocab → XPhoneBERT IPA tokens
YOUR_TO_IPA = {
    "a":"a",  "b":"b",  "d":"d",  "e":"e",  "f":"f",
    "i":"i",  "j":"j",  "k":"k",  "l":"l",  "m":"m",
    "n":"n",  "o":"o",  "p":"p",  "s":"s",  "t":"t",
    "u":"u",  "v":"v",  "w":"w",  "y":"y",  "z":"z",
    "ø":"ø",  "ŋ":"ŋ",  "ɔ":"ɔ",  "ə":"ə",  "ɛ":"ɛ",
    "ɡ":"ɡ",  "ɲ":"ɲ",  "ʁ":"ʁ",  "ʃ":"ʃ",  "ʒ":"ʒ",
    "dʒ":"ʒ", "tʃ":"ʃ", "ts":"s", "@":"ə",
    "§":None, "*":None, "[PAD]":None, "[UNK]":None,
}

# =========================
# Dataset (cached)
# =========================
if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset from disk...")
    dataset = load_from_disk(preprocessed_path)
else:
    print("Preprocessing from scratch...")

    dataset = load_dataset(
        "audiofolder",
        data_dir="/vol/experiments2/cbrazier/transcription/datasets/ester1_ester2_epac"
    )
    dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def is_valid(example):
        has_unk    = "|ø|n|k|" in example["phonemes"]
        too_short  = len(example["audio"]["array"]) < 24000
        few_phones = len(example["phoneme_single_v1"].split()) < 5
        return not has_unk and not too_short and not few_phones

    for split in ["train", "validation", "test"]:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(is_valid, num_proc=6)
        print(f"  {split}: {before} → {len(dataset[split])}")

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

    def preprocess_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
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
            preprocess_batch, batched=True, batch_size=32, num_proc=6,
            remove_columns=dataset[split].column_names,
            cache_file_name=f"/vol/experiments/cache_imbenamor/map_FS_{split}.arrow",
        )
    dataset.save_to_disk(preprocessed_path)

# Curriculum: shortest utterances first
lengths        = dataset["train"]["length"]
sorted_indices = np.argsort(lengths)
sorted_dataset = dataset["train"].select(sorted_indices)
print(f"Min length: {min(lengths)/16000:.1f}s  Max: {max(lengths)/16000:.1f}s")

# =========================
# Processor
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
# Data collator
# =========================
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
# Metrics
# =========================
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
    total_errors, total_ref = 0, 0
    for pred_id_seq, label_id_seq in zip(pred_ids, label_ids):
        pred_tokens = ctc_collapse(
            [processor.tokenizer.convert_ids_to_tokens(int(i)) for i in pred_id_seq]
        )
        label_tokens = [
            processor.tokenizer.convert_ids_to_tokens(int(i))
            for i in label_id_seq
            if i not in [-100, processor.tokenizer.pad_token_id]
        ]
        total_errors += editdistance.eval(pred_tokens, label_tokens)
        total_ref    += max(1, len(label_tokens))
    return {"PER": float(total_errors / total_ref)}

# =========================
# Loss weights
# IMPORTANT: λ_fs=0.15 (not 1.0) — prevents L_fs dominating encoder gradients
# =========================
def get_loss_weights(global_step: int, warmup_fs: int = 800, warmup_m: int = 5000):
    w_fs = min(1.0, global_step / max(warmup_fs, 1))
    w_m  = min(1.0, global_step / max(warmup_m,  1))
    return 5.0, 0.15 * w_fs, 0.5 * w_m


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
            idx    = torch.randint(0, T, (B,), device=X.device)
            frames = X[torch.arange(B, device=X.device), idx]
            embeddings.append(frames.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    num_vars   = model.quantizer.num_vars
    perm       = torch.randperm(embeddings.shape[0])[:num_vars]
    with torch.no_grad():
        model.quantizer.codevectors.copy_(embeddings[perm].to(device))
        model.quantizer.ema_embed.copy_(embeddings[perm].to(device))
        model.quantizer.cluster_size.fill_(1.0)
    print(f"Quantizer initialized from {embeddings.shape[0]} real frames")
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
        self.register_buffer("ema_embed",    torch.randn(num_vars, input_dim))
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
                self.cluster_size.mul_(self.decay).add_((1 - self.decay) * counts)
                self.ema_embed.mul_(self.decay).add_((1 - self.decay) * embed_sum)
                self.codevectors.copy_(
                    self.ema_embed / self.cluster_size.unsqueeze(1).clamp(min=1e-5)
                )
        else:
            indices = logits.argmax(dim=-1)
            probs   = F.one_hot(indices, self.num_vars).float()
        return torch.matmul(probs, self.codevectors), probs


# =========================
# Model
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Wav2Vec2ForCTC

# =========================
# IPA mapping (corpus labels → XPhoneBERT IPA)
# =========================
YOUR_TO_IPA = {
    "a":"a",  "b":"b",  "d":"d",  "e":"e",  "f":"f",
    "i":"i",  "j":"j",  "k":"k",  "l":"l",  "m":"m",
    "n":"n",  "o":"o",  "p":"p",  "s":"s",  "t":"t",
    "u":"u",  "v":"v",  "w":"w",  "y":"y",  "z":"z",
    "ø":"ø",  "ŋ":"ŋ",  "ɔ":"ɔ",  "ə":"ə",  "ɛ":"ɛ",
    "ɡ":"ɡ",  "ɲ":"ɲ",  "ʁ":"ʁ",  "ʃ":"ʃ",  "ʒ":"ʒ",
    "dʒ":"ʒ", "tʃ":"ʃ", "ts":"s", "@":"ə",
    "§":None, "*":None, "[PAD]":None, "[UNK]":None,
}


# =========================
# Gumbel Quantizer (unchanged)
# =========================
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
        self.register_buffer("ema_embed",    torch.randn(num_vars, input_dim))
        nn.init.uniform_(self.codevectors, -1.0, 1.0)

    def forward(self, X: torch.Tensor):
        B, T, D = X.shape
        logits  = self.weight_proj(X)
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temp, hard=True)
            with torch.no_grad():
                flat_probs = probs.reshape(-1, self.num_vars)
                flat_X     = X.reshape(-1, D)
                counts     = flat_probs.sum(0)
                embed_sum  = flat_probs.T @ flat_X
                self.cluster_size.mul_(self.decay).add_((1 - self.decay) * counts)
                self.ema_embed.mul_(self.decay).add_((1 - self.decay) * embed_sum)
                self.codevectors.copy_(
                    self.ema_embed / self.cluster_size.unsqueeze(1).clamp(min=1e-5)
                )
        else:
            indices = logits.argmax(dim=-1)
            probs   = F.one_hot(indices, self.num_vars).float()
        return torch.matmul(probs, self.codevectors), probs


# =========================
# Model v2: detached CTC pooling + XPhoneBERT + positional encoding
#
# KEY CHANGES vs previous XPhoneBERT model:
#   1. X.detach() + logits.detach() in CTC pooling   → encoder stability
#   2. pos_embed (learnable, LR=1e-3)                → breaks oscillations
#   3. align_temperature = 20.0  (was 15.0)          → sharper columns
#   4. gaussian = -50, ahead = -60  (was -30/-40)    → stronger prior
#   5. Encoder LR = 1e-7  (was 5e-6)                → prevents drift
# =========================
class Wav2Vec2ForCTC_FS_REC(nn.Module):
    """
    Wav2Vec2 + detached CTC pooling + XPhoneBERT + positional encoding.

    Phoneme embeddings:
        Y_ctc  = CTC_pool(X.detach(), logits.detach())   acoustic, stable
        Y_bert = bert_proj(XPhoneBERT(ipa_seq))          contextual, frozen BERT
        pos    = pos_embed(positions)                     positional, learnable
        Y_emb  = fy(cat([Y_ctc, Y_bert + pos]))          fused

    Why detach:
        Keeps encoder CTC posteriors stable throughout training.
        Prevents attractor bands that appeared at epoch 16+ in previous run.

    Why pos_embed:
        XPhoneBERT gives similar embeddings for adjacent same-phoneme
        occurrences → argmax oscillates between rows → false boundaries.
        A small learnable position offset makes each position unique,
        reducing oscillations without losing contextual information.

    Why stronger prior (-50/-60 vs -30/-40):
        Reduces competition between adjacent rows further.
        Combined with pos_embed, should eliminate most oscillations.
    """

    def __init__(
        self,
        base_model:    Wav2Vec2ForCTC,
        vocab_size:    int,
        hidden_dim:    int,
        ctc_tokenizer,
        n_negatives:   int   = 50,
        temperature:   float = 0.1,
    ):
        super().__init__()

        self.wav2vec2      = base_model.wav2vec2
        self.ctc_head      = base_model.lm_head
        self.ctc_tokenizer = ctc_tokenizer
        self.n_negatives   = n_negatives
        self.temperature   = temperature
        self._keys_to_ignore_on_save = None

        # ── Alignment temperature + prior strengths ───────────────────────
        self.align_temperature = 20.0   # was 15.0 — sharper softmax
        self.diag_strength     = 50.0   # was 30.0 — stronger Gaussian
        self.ahead_strength    = 60.0   # was 40.0 — stronger one-sided

        # ── Alignment projections ─────────────────────────────────────────
        self.fx = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.fx.weight)

        # ── XPhoneBERT — frozen contextual encoder ────────────────────────
        print("Loading XPhoneBERT...")
        self.phoneme_bert   = AutoModel.from_pretrained("vinai/xphonebert-base")
        self.xphonebert_tok = AutoTokenizer.from_pretrained(
            "vinai/xphonebert-base", add_prefix_space=True
        )
        for p in self.phoneme_bert.parameters():
            p.requires_grad = False

        # bert_proj: 768 → hidden_dim, trainable (LR=5e-4)
        self.bert_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
        )

        # ── Positional encoding (NEW) ─────────────────────────────────────
        # Learnable position embedding added to Y_bert before fusion.
        # Makes each phoneme position unique even if XPhoneBERT embeddings
        # are similar → reduces oscillations in argmax path.
        # Small init (std=0.01) so it does not dominate XPhoneBERT signal.
        # LR=1e-3 so it adapts quickly.
        self.pos_embed = nn.Embedding(512, hidden_dim)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.01)

        # ── Fusion: cat([Y_ctc, Y_bert + pos]) → hidden_dim ──────────────
        # fy input: 2 * hidden_dim
        # Initialized so both Y_ctc and Y_bert contribute equally at step 0
        self.fy = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        W = torch.zeros(hidden_dim, 2 * hidden_dim)
        W[:, :hidden_dim] = torch.eye(hidden_dim) * 0.5   # Y_ctc half
        W[:, hidden_dim:] = torch.eye(hidden_dim) * 0.5   # Y_bert half
        self.fy.weight.data.copy_(W)

        # ── Quantizer + reconstruction ────────────────────────────────────
        self.quantizer           = GumbelQuantizerEMA(input_dim=hidden_dim, decay=0.99)
        self.reconstruction_head = nn.Linear(2 * hidden_dim, hidden_dim)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.wav2vec2.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.wav2vec2.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # CTC greedy decode
    # ------------------------------------------------------------------
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
    # IPA conversion
    # ------------------------------------------------------------------
    def decode_to_ipa(self, labels_clean: torch.Tensor) -> list:
        ipa_sequences = []
        for seq in labels_clean:
            ipa_phones = []
            for token_id in seq:
                tid = token_id.item()
                if tid == 0:
                    break
                corpus_label = self.ctc_tokenizer.convert_ids_to_tokens(tid)
                ipa          = YOUR_TO_IPA.get(corpus_label, None)
                if ipa is not None:
                    ipa_phones.append(ipa)
            if not ipa_phones:
                ipa_phones = ["ə"]
            ipa_sequences.append(ipa_phones)
        return ipa_sequences

    # ------------------------------------------------------------------
    # Detached CTC pooling  ← KEY CHANGE: X.detach() + logits.detach()
    # ------------------------------------------------------------------
    def compute_ctc_embeddings(self, X_det, logits_det, labels_clean):
        """
        Type-level phoneme embeddings via CTC posterior pooling.
        X and logits are DETACHED — encoder is not updated by alignment
        losses, preventing the drift that caused attractor bands in v1.
        """
        B, T, D    = X_det.shape
        V          = logits_det.shape[-1]
        ctc_probs  = logits_det.softmax(dim=-1)
        token_mass = ctc_probs.sum(dim=1).clamp(min=1e-6)
        token_bank = torch.einsum("btd,btv->bvd", X_det, ctc_probs)
        token_bank = token_bank / token_mass.unsqueeze(-1)

        Y_ctc_list = []
        for b in range(B):
            seq = labels_clean[b]
            seq = seq[seq != 0]
            if seq.numel() == 0:
                seq = torch.tensor([1], device=X_det.device, dtype=torch.long)
            seq = seq.clamp(0, V - 1)
            Y_ctc_list.append(token_bank[b].index_select(0, seq))

        return torch.nn.utils.rnn.pad_sequence(
            Y_ctc_list, batch_first=True, padding_value=0.0
        )   # (B, N, D)

    # ------------------------------------------------------------------
    # XPhoneBERT with BPE aggregation + chunking for long sequences
    # ------------------------------------------------------------------
    def _bert_encode_chunk(self, ipa_seq: list, device) -> torch.Tensor:
        enc    = self.xphonebert_tok(
            [ipa_seq], return_tensors="pt",
            is_split_into_words=True, padding=False,
        ).to(device)
        with torch.no_grad():
            out = self.phoneme_bert(**enc)
        hidden   = out.last_hidden_state[0]
        word_ids = enc.word_ids(batch_index=0)
        embs = []
        for n in range(len(ipa_seq)):
            positions = [i for i, w in enumerate(word_ids) if w == n]
            if positions:
                embs.append(hidden[positions, :].mean(dim=0))
            else:
                embs.append(torch.zeros(768, device=device))
        return torch.stack(embs)   # (N, 768)

    def _bert_forward_single(self, ipa_seq: list, device) -> torch.Tensor:
        MAX_BPE   = 510
        enc_check = self.xphonebert_tok(
            [ipa_seq], is_split_into_words=True, add_special_tokens=False
        )
        if len(enc_check["input_ids"][0]) <= MAX_BPE:
            return self._bert_encode_chunk(ipa_seq, device)

        # Binary search chunking for sequences > 510 BPE tokens
        N_phones    = len(ipa_seq)
        phone_embs  = [None] * N_phones
        chunk_start = 0
        while chunk_start < N_phones:
            lo, hi = 1, N_phones - chunk_start
            while lo < hi:
                mid = (lo + hi + 1) // 2
                sub = ipa_seq[chunk_start : chunk_start + mid]
                enc = self.xphonebert_tok(
                    [sub], is_split_into_words=True, add_special_tokens=False
                )
                if len(enc["input_ids"][0]) <= MAX_BPE:
                    lo = mid
                else:
                    hi = mid - 1
            chunk_end = chunk_start + lo
            chunk_emb = self._bert_encode_chunk(ipa_seq[chunk_start:chunk_end], device)
            for i, emb in enumerate(chunk_emb):
                phone_embs[chunk_start + i] = emb
            chunk_start = chunk_end
        return torch.stack(phone_embs)

    def compute_bert_embeddings(self, labels_clean, device):
        ipa_sequences = self.decode_to_ipa(labels_clean)
        all_embs = []
        for ipa_seq in ipa_sequences:
            if len(ipa_seq) == 0:
                all_embs.append(torch.zeros(1, 768, device=device))
                continue
            all_embs.append(self._bert_forward_single(ipa_seq, device))
        return torch.nn.utils.rnn.pad_sequence(
            all_embs, batch_first=True, padding_value=0.0
        )   # (B, N_max, 768)

    # ------------------------------------------------------------------
    # Fused phoneme embeddings  ← pos_embed added to Y_bert (NEW)
    # ------------------------------------------------------------------
    def compute_phoneme_embeddings(self, X_det, logits_det, labels_clean, device):
        B = X_det.shape[0]

        # Acoustic embeddings — detached, stable
        Y_ctc  = self.compute_ctc_embeddings(X_det, logits_det, labels_clean)

        # Contextual embeddings — frozen BERT + trainable bert_proj
        H_bert = self.compute_bert_embeddings(labels_clean, device).to(device)
        Y_bert = self.bert_proj(H_bert)   # (B, N, D)

        # Align N dimension
        N      = Y_ctc.shape[1]
        N_bert = Y_bert.shape[1]
        if N_bert >= N:
            Y_bert = Y_bert[:, :N, :]
        else:
            pad    = torch.zeros(B, N - N_bert, Y_ctc.shape[-1],
                                 device=device, dtype=Y_bert.dtype)
            Y_bert = torch.cat([Y_bert, pad], dim=1)

        # Positional encoding — makes each position unique (NEW)
        # Clamp to 511 in case N > 512 (extremely long utterances)
        positions = torch.arange(N, device=device).clamp(max=511)
        pos       = self.pos_embed(positions).unsqueeze(0)   # (1, N, D)
        Y_bert    = Y_bert + pos                              # unique per position

        # Fuse acoustic + contextual
        return self.fy(torch.cat([Y_ctc, Y_bert], dim=-1))   # (B, N, D)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def mask_features(self, X: torch.Tensor, p_low: float = 0.1, p_high: float = 0.5):
        B, T, _ = X.shape
        mask_prob = torch.FloatTensor(1).uniform_(p_low, p_high).item()
        mask      = torch.rand(B, T, device=X.device) < mask_prob
        X_masked  = X.clone()
        X_masked[mask] = 0.0
        return X_masked, mask

    # ------------------------------------------------------------------
    # Alignment with stronger prior  ← diag_strength=50, ahead=60
    # ------------------------------------------------------------------
    def compute_alignment(self, X: torch.Tensor, Y_emb: torch.Tensor) -> torch.Tensor:
        B, T, D = X.shape
        _, N, _ = Y_emb.shape

        X_proj = self.fx(X)
        X_norm = F.normalize(X_proj, dim=-1)
        Y_norm = F.normalize(Y_emb,  dim=-1)

        D_mat  = self.align_temperature * torch.matmul(
            Y_norm, X_norm.transpose(1, 2)
        )   # (B, N, T)

        n_idx = torch.arange(N, device=X.device).float() / max(N - 1, 1)
        t_idx = torch.arange(T, device=X.device).float() / max(T - 1, 1)
        diff          = n_idx.unsqueeze(1) - t_idx.unsqueeze(0)
        gaussian      = -self.diag_strength  * diff ** 2
        ahead_penalty = -self.ahead_strength * torch.clamp(diff, min=0) ** 2
        prior         = gaussian + ahead_penalty
        D_mat         = D_mat + prior.unsqueeze(0)

        return torch.softmax(D_mat, dim=1)

    # ------------------------------------------------------------------
    # Forward-sum loss (unchanged)
    # ------------------------------------------------------------------
    def forward_sum_loss(self, A, label_lengths, input_lengths):
        B, N, T      = A.shape
        blank_prob   = torch.full((B, 1, T), 1e-8, device=A.device, dtype=torch.float32)
        A_with_blank = torch.cat([blank_prob, A.float()], dim=1)
        log_probs    = torch.log(A_with_blank.clamp(min=1e-8)).permute(2, 0, 1)
        labels = torch.zeros(B, N, dtype=torch.long, device=A.device)
        for b in range(B):
            n = label_lengths[b].item()
            labels[b, :n] = torch.arange(1, n + 1, device=A.device)
        loss = F.ctc_loss(
            log_probs, labels, input_lengths, label_lengths,
            blank=0, reduction="none", zero_infinity=True,
        )
        return (loss / label_lengths.float().clamp(min=1)).mean()

    # ------------------------------------------------------------------
    # Contrastive reconstruction loss (unchanged)
    # ------------------------------------------------------------------
    def contrastive_loss(self, H, X_quantized, mask, temperature=None):
        B, T, D   = H.shape
        temp      = temperature if temperature is not None else self.temperature
        H_flat    = H.reshape(B * T, D)
        Q_flat    = X_quantized.reshape(B * T, D)
        mask_flat = mask.reshape(B * T)
        if mask_flat.sum() == 0:
            return H_flat.sum() * 0.0
        H_norm   = F.normalize(H_flat + 1e-8, dim=-1)
        Q_norm   = F.normalize(Q_flat + 1e-8, dim=-1)
        masked_H = H_norm[mask_flat]
        masked_Q = Q_norm[mask_flat]
        M        = masked_H.shape[0]
        n_neg    = min(self.n_negatives, B * T - 1)
        neg_idx  = torch.randint(0, B * T, (M, n_neg), device=H.device)
        Q_neg    = Q_norm[neg_idx]
        sim_pos  = (masked_H * masked_Q).sum(dim=-1) / temp
        sim_neg  = torch.bmm(Q_neg, masked_H.unsqueeze(-1)).squeeze(-1) / temp
        all_sims = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        return (-sim_pos + torch.logsumexp(all_sims, dim=1)).mean()

    # ------------------------------------------------------------------
    # Forward  ← X.detach() + logits.detach() passed to CTC pooling
    # ------------------------------------------------------------------
    def forward(self, input_values, attention_mask=None, labels=None, global_step=0):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        X       = outputs.last_hidden_state   # (B, T, D)
        logits  = self.ctc_head(X)            # (B, T, vocab)

        with torch.no_grad():
            labels_clean = self._ctc_decode_batch(logits)

        # Detached CTC pooling — encoder not updated by alignment losses
        Y_emb = self.compute_phoneme_embeddings(
            X.detach(), logits.detach(), labels_clean, device=X.device
        )

        A = self.compute_alignment(X, Y_emb)

        if labels is None:
            return {"loss": None, "logits": logits, "alignment": A}

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

        loss_ctc = F.ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            labels, input_lengths, label_lengths,
            blank=0, reduction="mean", zero_infinity=True,
        )

        loss_fs = self.forward_sum_loss(A, label_lengths_pred, input_lengths)

        X_quantized, _  = self.quantizer(X.detach())
        X_masked, mask  = self.mask_features(X)
        YA              = torch.matmul(A.transpose(1, 2), Y_emb)
        H               = self.reconstruction_head(torch.cat([X_masked, YA], dim=-1))
        current_temp    = max(0.1, 1.0 - 0.9 * min(1.0, global_step / 10000))
        self.quantizer.temp = current_temp
        loss_m = self.contrastive_loss(H, X_quantized, mask, temperature=0.1)

        return {
            "loss":     loss_ctc * 5.0 + loss_fs * 0.15 + loss_m * 0.5,
            "logits":   logits,
            "alignment": A,
            "loss_ctc": loss_ctc.detach(),
            "loss_fs":  loss_fs.detach(),
            "loss_m":   loss_m.detach(),
        }

# =========================
# Load model
# =========================
base_model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction = "mean",
    ctc_zero_infinity  = True,
    pad_token_id       = processor.tokenizer.pad_token_id,
    vocab_size         = len(processor.tokenizer),
)

model = Wav2Vec2ForCTC_FS_REC(
    base_model    = base_model,
    vocab_size    = len(processor.tokenizer),
    hidden_dim    = base_model.config.hidden_size,
    ctc_tokenizer = processor.tokenizer,
    n_negatives   = 50,
    temperature   = 0.1,
)
model.wav2vec2.feature_extractor._freeze_parameters()

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir                  = output_dir,
    group_by_length             = False,
    length_column_name          = "length",
    per_device_train_batch_size = 4,    # reduced: XPhoneBERT on GPU needs memory
    gradient_accumulation_steps = 64,   # effective batch = 256
    lr_scheduler_type           = "constant_with_warmup",
    warmup_steps                = 200,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
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


# =========================
# Trainer
# =========================
class FSCTCTrainer(Trainer):

    def __init__(self, *args, log_every: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_every    = log_every
        self._step_count   = 0
        self._epoch_losses = {"ctc": [], "fs": [], "m": []}

    def _get_eval_sampler(self, eval_dataset):
        return None

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_values   = inputs["input_values"],
            attention_mask = inputs.get("attention_mask"),
            labels         = inputs["labels"],
            global_step    = self.state.global_step,
        )
        loss = outputs["loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ✗ NaN/Inf loss at step {self.state.global_step}")
            return (loss * 0.0, outputs) if return_outputs else loss * 0.0

        loss_ctc = outputs.get("loss_ctc")
        loss_fs  = outputs.get("loss_fs")
        loss_m   = outputs.get("loss_m")

        if loss_ctc is not None:
            self._epoch_losses["ctc"].append(loss_ctc.item())
            self._epoch_losses["fs"].append(loss_fs.item())
            self._epoch_losses["m"].append(loss_m.item())

        self._step_count += 1
        if self._step_count % self._log_every == 0 and loss_ctc is not None:
            self._log_loss_components(loss, loss_ctc, loss_fs, loss_m, inputs, outputs)

        return (loss, outputs) if return_outputs else loss

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self._epoch_losses["ctc"]:
            return
        mean_ctc = np.mean(self._epoch_losses["ctc"])
        mean_fs  = np.mean(self._epoch_losses["fs"])
        mean_m   = np.mean(self._epoch_losses["m"])
        print(f"[epoch {int(state.epoch)}] L_ctc={mean_ctc:.4f}  L_fs={mean_fs:.4f}  L_m={mean_m:.4f}")
        self.log({"epoch/loss_ctc": mean_ctc, "epoch/loss_fs": mean_fs, "epoch/loss_m": mean_m})
        self._epoch_losses = {"ctc": [], "fs": [], "m": []}

    def _log_loss_components(self, loss, loss_ctc, loss_fs, loss_m, inputs, outputs):
        effective_step    = self.state.global_step + RESUME_STEP
        λ_ctc, λ_fs, λ_m = get_loss_weights(effective_step)

        with torch.no_grad():
            logits         = outputs["logits"]
            post_sharpness = logits.softmax(dim=-1).max(dim=-1).values.mean().item()

        print(
            f"[step {self._step_count}] "
            f"loss={loss.item():.4f}  "
            f"L_ctc={loss_ctc.item():.4f} (w={λ_ctc:.1f})  "
            f"L_fs={loss_fs.item():.4f} (w={λ_fs:.2f})  "
            f"L_m={loss_m.item():.4f} (w={λ_m:.2f})  "
            f"posterior_sharpness={post_sharpness:.3f}"
        )
        self.log({
            "train/loss_ctc":            loss_ctc.item(),
            "train/loss_fs":             loss_fs.item(),
            "train/loss_m":              loss_m.item(),
            "train/weighted_ctc":        (λ_ctc * loss_ctc).item(),
            "train/weighted_fs":         (λ_fs  * loss_fs).item(),
            "train/weighted_m":          (λ_m   * loss_m).item(),
            "train/posterior_sharpness": post_sharpness,
        })

        if loss_ctc.item() > 0.5:
            print("  ⚠  L_ctc rising — possible encoder drift from L_fs gradients")
        if post_sharpness < 0.5:
            print("  ⚠  Posterior sharpness low — encoder drifting")
        if torch.isnan(loss):
            print("  ✗  NaN in total loss")

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {"params": list(self.model.wav2vec2.parameters()),              "lr": 1e-7},
            {"params": list(self.model.ctc_head.parameters()),              "lr": 1e-7},
            {"params": list(self.model.fx.parameters()),                    "lr": 1e-4},
            {"params": list(self.model.fy.parameters()),                    "lr": 1e-4},
            {"params": list(self.model.bert_proj.parameters()),             "lr": 5e-4},
            {"params": list(self.model.pos_embed.parameters()),             "lr": 1e-3},
            {"params": list(self.model.reconstruction_head.parameters()),   "lr": 1e-4},
            {"params": list(self.model.quantizer.weight_proj.parameters()), "lr": 1e-3},
        ], weight_decay=self.args.weight_decay)
        return self.optimizer

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self._step_count % self._log_every == 0:
            norms = {}
            for name, module in [
                ("encoder",    model.wav2vec2.encoder),
                ("ctc_head",   model.ctc_head),
                ("fx",         model.fx),
                ("fy",         model.fy),
                ("bert_proj",  model.bert_proj),
                ("recon_head", model.reconstruction_head),
            ]:
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
                out          = model.wav2vec2(
                    batch["input_values"],
                    attention_mask=batch.get("attention_mask")
                )
                X            = out.last_hidden_state
                logits       = model.ctc_head(X)
                labels_clean = model._ctc_decode_batch(logits)
                Y_emb        = model.compute_phoneme_embeddings(
                    X, logits, labels_clean, X.device
                )
                A            = model.compute_alignment(X, Y_emb)
                self._plot_alignment_to_tensorboard(A, labels_clean, epoch=int(self.state.epoch))

                eps       = 1e-8
                entropy   = -(A * (A + eps).log()).sum(dim=1)
                mean_H    = entropy.mean().item()
                max_H     = np.log(labels_clean.shape[1] + eps)
                sharpness = 100 * (1 - mean_H / max_H)

            print(f"\n[alignment @ epoch {int(self.state.epoch)}] sharpness: {sharpness:.1f}%\n")
            self.log({"alignment_sharpness": sharpness})

        except Exception as e:
            print(f"Alignment sharpness logging failed: {e}")
        finally:
            model.train()
        return sharpness

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(
                input_values   = inputs["input_values"],
                attention_mask = inputs.get("attention_mask"),
                labels         = inputs["labels"],
            )
        if prediction_loss_only:
            return (outputs["loss"], None, None)
        return (outputs["loss"], outputs["logits"], inputs["labels"])

    def _plot_alignment_to_tensorboard(self, A, labels_clean, epoch):
        writer   = SummaryWriter(log_dir=self.args.logging_dir)
        n_phones = (labels_clean[0] != 0).sum().item()
        A_sample = A[0, :n_phones, :].cpu().float().numpy()
        fig, ax  = plt.subplots(figsize=(12, 4))
        im       = ax.imshow(A_sample, aspect="auto", origin="lower",
                             cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=ax)
        path = A[0, :n_phones, :].argmax(dim=0).cpu().numpy()
        ax.plot(range(len(path)), path, color="lime", linewidth=1.0, label="argmax path")
        T = A_sample.shape[1]
        ax.plot(range(T), np.linspace(0, n_phones - 1, T),
                color="deepskyblue", linewidth=1.0, linestyle="--", label="ideal diagonal")
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

trainer.train()

logs_path = os.path.join(output_dir, "logs.json")
with open(logs_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)
print(f"Training complete. Logs saved to {logs_path}")