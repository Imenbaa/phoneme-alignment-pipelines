import os
import json
import pickle
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import tgt
from collections import Counter

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
sampa_to_api_single = {
    'a': 'a', 'e': 'e','i': 'i','o': 'o','u': 'u','y': 'y','2': 'ø','9': '9','@': 'ə','E': 'ɛ','O': 'ɔ','a~': '@','e~': '5','9~': '1','o~': '§','b': 'b','d': 'd','f': 'f','g': 'ɡ','k': 'k','l': 'l','m': 'm','n': 'n','n=':'n','p': 'p','t': 't','v': 'v','w': 'w','z': 'z','j': 'j','R': 'ʁ','N': 'ŋ','H': 'ɥ','J': 'ɲ','S': 'ʃ','Z=': 'dʒ','s': 'ts','Z': 'dʒ','m=': 'm','_': '_','spn': 'spn','unk': 'spn',"%":'spn',"?":"spn","0":"spn"}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    def __init__(self, base_model, vocab_size, hidden_dim,
                 ctc_tokenizer, n_negatives=50, temperature=0.1):
        super().__init__()
        self.wav2vec2            = base_model.wav2vec2
        self.ctc_head            = base_model.lm_head
        self.ctc_tokenizer       = ctc_tokenizer
        self.n_negatives         = n_negatives
        self.temperature         = temperature
        self.align_temperature   = 5.0                              # ← must match training
        self.fx                  = nn.Linear(hidden_dim, hidden_dim, bias=False)  # ← no bias
        self.quantizer           = GumbelQuantizerEMA(input_dim=hidden_dim)
        self.reconstruction_head = nn.Linear(2 * hidden_dim, hidden_dim)

    def _ctc_decode_batch(self, logits):
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

    def compute_phoneme_embeddings_from_ctc(self, X, logits, labels_clean):
        B, T, D   = X.shape
        ctc_probs = logits.softmax(dim=-1)
        Y_emb_list = []
        for b in range(B):
            phone_embs = []
            for p_id in labels_clean[b]:
                p_id = p_id.item()
                if p_id == 0:
                    phone_embs.append(torch.zeros(D, device=X.device))
                    continue
                weights = ctc_probs[b, :, p_id]
                weight_sum = weights.sum()
                if weight_sum < 1e-6:
                    weights = torch.ones_like(weights) / weights.shape[0]
                else:
                    weights = weights / weight_sum
                emb = (weights.unsqueeze(-1) * X[b]).sum(0)
                phone_embs.append(emb)
            Y_emb_list.append(torch.stack(phone_embs))
        return torch.stack(Y_emb_list)

    def compute_alignment(self, X, Y_emb):
        B, T, D = X.shape
        _, N, _ = Y_emb.shape
        X_proj  = self.fx(X)
        X_norm  = F.normalize(X_proj, dim=-1)
        Y_norm  = F.normalize(Y_emb,  dim=-1)
        D_mat   = self.align_temperature * torch.matmul(
            Y_norm, X_norm.transpose(1, 2)
        )
        n_idx = torch.arange(N, device=X.device).float() / max(N - 1, 1)
        t_idx = torch.arange(T, device=X.device).float() / max(T - 1, 1)
        prior = -10.0 * (n_idx.unsqueeze(1) - t_idx.unsqueeze(0)) ** 2
        D_mat = D_mat + prior.unsqueeze(0)
        return torch.softmax(D_mat, dim=1)

    def forward(self, input_values, attention_mask=None):
        outputs      = self.wav2vec2(input_values, attention_mask=attention_mask)
        X            = outputs.last_hidden_state
        logits       = self.ctc_head(X)
        with torch.no_grad():
            labels_clean = self._ctc_decode_batch(logits)
        Y_emb = self.compute_phoneme_embeddings_from_ctc(
            X.detach(), logits.detach(), labels_clean
        )
        A = self.compute_alignment(X, Y_emb)
        return logits, A, labels_clean

with open("vocab_w2vCTC.json") as f:
    vocab = json.load(f)
def map_ref_to_api_single(ph, sampa_to_api_single):
     
    if ph in sampa_to_api_single:
        mapped = sampa_to_api_single[ph]
        return mapped
    else:
        if ph!= None and ph != "":
            return ph

def read_textgrid(tg_path, tier_name="phone"):
    """
    Returns list of {'phoneme': str, 'start': float, 'end': float}.
    Skips silence intervals (empty label, SIL, sil, sp).
    """
    tg        = tgt.io.read_textgrid(tg_path)
    tier      = tg.get_tier_by_name(tier_name)
    silence   = {"", "SIL", "sil", "spn", "SP", "<SIL>", "_","0","fe~","sjo~","Ra~"}
    intervals = []
    for iv in tier.intervals:
        label = iv.text.strip()
        phoneme=map_ref_to_api_single(label,sampa_to_api_single)
        if phoneme in silence:
            continue
        intervals.append({
            "phoneme": phoneme,
            "start":   round(iv.start_time, 6),
            "end":     round(iv.end_time,   6),
        })
        if phoneme==None:
            print(phoneme)
    return intervals

def extract_intervals_forced(logits, A, tokenizer, duration_sec, frame_shift=0.02):
    """
    Produces EXACTLY the same phoneme sequence as CTC decoding.
    Uses A only to find boundaries — every CTC phoneme is guaranteed
    to appear in the output even if alignment skips it.
    """
    blank_id = tokenizer.pad_token_id

    # ── Step 1: CTC sequence — authoritative labels ──────────────────
    pred_ids  = logits[0].argmax(dim=-1).tolist()
    collapsed = []
    prev = None
    for t in pred_ids:
        if t != prev:
            if t != blank_id:
                collapsed.append(t)
        prev = t

    if not collapsed:
        return []

    N    = len(collapsed)
    A_np = A[0, :N, :].cpu().float().numpy()  # (N, T) — trim to CTC length
    T    = A_np.shape[1]

    # ── Step 2: Find boundaries using A ─────────────────────────────
    # Boundary between phoneme n and n+1 = first frame where
    # A[n+1, t] > A[n, t] after the previous boundary.
    # If no crossing found, place boundary proportionally.
    boundaries = [0]

    for n in range(N - 1):
        prev_b = boundaries[-1]
        remaining = N - n - 1          # phonemes still to place after n+1

        # Score difference: positive where n+1 is stronger than n
        diff      = A_np[n+1, prev_b:] - A_np[n, prev_b:]
        crossings = np.where(diff > 0)[0]

        if len(crossings) > 0:
            boundary = prev_b + int(crossings[0])
        else:
            # No crossing — divide remaining frames equally
            boundary = prev_b + max(1, (T - prev_b) // (remaining + 1))

        # Ensure there is always room for remaining phonemes
        boundary = min(boundary, T - remaining)
        boundary = max(boundary, prev_b + 1)
        boundaries.append(boundary)

    boundaries.append(T)

    # ── Step 3: Build intervals ──────────────────────────────────────
    silence   = {"[PAD]", "[UNK]", ""}
    intervals = []

    for n in range(N):
        phoneme_str = tokenizer.convert_ids_to_tokens(collapsed[n])
        if phoneme_str not in silence:
            intervals.append({
                "phoneme": phoneme_str,
                "start":   round(boundaries[n]     * frame_shift, 6),
                "end":     round(boundaries[n + 1] * frame_shift, 6),
            })

    return intervals
def align_long_audio(model, audio, tokenizer, feature_extractor,
                     device, max_duration=30.0, frame_shift=0.02):
    """
    Segment audio into chunks of max_duration seconds,
    run alignment on each, then concatenate intervals.
    """
    sr           = 16000
    chunk_size   = int(max_duration * sr)
    total_frames = len(audio)
    all_intervals = []
    offset_sec   = 0.0

    start = 0
    while start < total_frames:
        end   = min(start + chunk_size, total_frames)
        chunk = audio[start:end]
        dur   = len(chunk) / sr

        inputs = feature_extractor(
            chunk, sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True,
        )
        input_values   = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            logits, A, labels_clean = model(input_values, attention_mask)

        intervals = extract_intervals_forced(
    logits, A, tokenizer, dur, frame_shift
)

        # Shift intervals by current offset
        for iv in intervals:
            all_intervals.append({
                "phoneme": iv["phoneme"],
                "start":   round(iv["start"] + offset_sec, 6),
                "end":     round(iv["end"]   + offset_sec, 6),
            })

        offset_sec += dur
        start      += chunk_size

    return all_intervals
checkpoint = "results/w2vCTC_joint_nofxfy/checkpoint-21092"
ctc_checkpoint = "results/w2vCTC/checkpoint-23430"
audio_dir = "/vol/corpora/Rhapsodie/wav16k_corrected"
textgrid_dir = "/vol/corpora/Rhapsodie/TextGrids-fev2013/"
vocab = "vocab_w2vCTC.json"
output="results/rhap_ctc_FS.pkl"
tier = "phone"
device = "cpu"
print(f"Device: {device}")

# ── Processor ─────────────────────────────────────────────────────────
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file           = vocab,
    unk_token            = "[UNK]",
    pad_token            = "[PAD]",
    word_delimiter_token = "",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size         = 1,
    sampling_rate        = 16000,
    padding_value        = 0.0,
    do_normalize         = True,
    return_attention_mask = True,
)
print("Loading processor...")
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
)
# ── Model ─────────────────────────────────────────────────────────────
print(f"Loading CTC architecture from {ctc_checkpoint}...")
base_model = Wav2Vec2ForCTC.from_pretrained(
    ctc_checkpoint,
    ctc_loss_reduction = "mean",
    ctc_zero_infinity  = True,
    pad_token_id       = tokenizer.pad_token_id,
    vocab_size         = len(tokenizer),
)

model = Wav2Vec2ForCTC_FS_REC(
    base_model,
    vocab_size = len(processor.tokenizer),
    hidden_dim = base_model.config.hidden_size,
    ctc_tokenizer = processor.tokenizer,
)
model.wav2vec2.feature_extractor._freeze_parameters()

bin_path = os.path.join(checkpoint, "pytorch_model.bin")
sft_path = os.path.join(checkpoint, "model.safetensors")
if os.path.exists(bin_path):
    state_dict = torch.load(bin_path, map_location="cpu")
elif os.path.exists(sft_path):
    from safetensors.torch import load_file
    state_dict = load_file(sft_path)
else:
    raise FileNotFoundError(
        f"No model weights found in {checkpoint}\n"
        f"Expected pytorch_model.bin or model.safetensors"
    )

missing, unexpected = model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
print("Model ready.")

audio_files = sorted(
    glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
)
results  = {}
n_done   = 0


for audio_path in audio_files:
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    # Match TextGrid
    tg_path = os.path.join(textgrid_dir, filename + "-Pro.TextGrid")

    # Reference intervals from TextGrid

    ref_intervals = read_textgrid(tg_path,tier)

    # Load audio
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_sec = len(audio) / 16000.0

    # Feature extraction
    inputs = feature_extractor(
        audio,
        sampling_rate        = 16000,
        return_tensors       = "pt",
        return_attention_mask = True,
    )
    input_values   = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Inference
    
    hyp_intervals = align_long_audio(
    model, audio, tokenizer, feature_extractor,
    device, max_duration=10.0
)
    

    results[filename] = {
        "file":          os.path.basename(audio_path),
        "ref_intervals": ref_intervals,
        "hyp_intervals": hyp_intervals,
    }

    n_done += 1
    if n_done % 50 == 0 or n_done == 1:
        print(f"  [{n_done}/{len(audio_files)}] {filename}: "
              f"ref={len(ref_intervals)} hyp={len(hyp_intervals)}")

# ── Save ───────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(output))
os.makedirs(out_dir, exist_ok=True)
with open(output, "wb") as f:
    pickle.dump(results, f)
print(f"Saved → {output}")

# Sanity print
if results:
    key    = next(iter(results))
    sample = results[key]
    print(f"\nSample entry ({key}):")
    print(f"  file:  {sample['file']}")
    print(f"  ref:   {sample['ref_intervals'][:2]}")
    print(f"  hyp:   {sample['hyp_intervals'][:2]}")
ref_phones = set()
hyp_phones = set()
for v in results.values():
    ref_phones.update(iv["phoneme"] for iv in v["ref_intervals"])
    hyp_phones.update(iv["phoneme"] for iv in v["hyp_intervals"])

print("REF:", sorted(ref_phones))
print("HYP:", sorted(hyp_phones))
print("Missing from HYP:", sorted(ref_phones - hyp_phones))
import json, pickle

# Reference phonemes seen in TextGrids
with open("results/rhap_ctc_FS.pkl", "rb") as f:
    results = pickle.load(f)


    
    