"""
diagnose_model.py — Pre-inference analysis for Wav2Vec2ForCTCWithVAD
====================================================================
Run this BEFORE inference to verify:
  1. Training convergence (loss / PER curves from trainer_state.json)
  2. VAD bias effectiveness (SIL logit shift in silence vs speech frames)
  3. Token distribution (blank / SIL / phoneme split in greedy output)
  4. Test-set PER with vs without VAD bias
  5. Per-phoneme error rates

Usage:
    python diagnose_model.py

Adjust the CONFIG block at the top to match your paths.
"""

import os, json, sys
os.environ["HF_HOME"] = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"] = "/vol/experiments/cache_imbenamor/datasets"
os.environ["HF_DATASETS_OFFLINE"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import editdistance
from collections import defaultdict

# ──────────────────────────────────────────────
# CONFIG  ← adjust these
# ──────────────────────────────────────────────
EXP_NAME      = "w2vCTC_VAD"
OUTPUT_DIR    = f"results/{EXP_NAME}"
VOCAB_NAME    = f"vocab_{EXP_NAME}.json"
PREPROC_PATH  = f"{OUTPUT_DIR}/preprocessed"
DIAG_DIR      = f"{OUTPUT_DIR}/diagnostics"
N_SAMPLES     = 200     # samples for token-distribution & VAD analysis
VAD_THRESHOLD = 0.005   # must match training
ALPHA         = 1.0     # must match training
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────

os.makedirs(DIAG_DIR, exist_ok=True)
class Wav2Vec2ForCTCWithVAD(Wav2Vec2ForCTC):

    def apply_vad_bias(self, logits, vad):
        vad = F.interpolate(vad.unsqueeze(1), size=logits.shape[1], mode="nearest").squeeze(1)
        vad = vad.clamp(1e-4, 1 - 1e-4)

        V = logits.shape[-1]

        sil_mask = torch.zeros(V, device=logits.device)
        sil_mask[self.config.sil_token_id] = 1.0
        sil_mask = sil_mask.view(1, 1, V)

        phoneme_mask = 1 - sil_mask
        vad = vad.unsqueeze(-1)

        logits = logits + (
            sil_mask * torch.log(1 - vad) +
            phoneme_mask * torch.log(vad)
        )
        return logits

    def forward(self, input_values, vad=None):
        outputs = self.wav2vec2(input_values)
        logits = self.lm_head(outputs.last_hidden_state)

        if vad is not None:
            logits = self.apply_vad_bias(logits, vad)

        return {"logits": logits}
# ══════════════════════════════════════════════
# 1. TRAINING CURVES
# ══════════════════════════════════════════════
def plot_training_curves():
    state_path = f"results/{EXP_NAME}/"+ "checkpoint-15625/trainer_state.json"
    if not os.path.exists(state_path):
        print("[SKIP] trainer_state.json not found — skipping curve plot.")
        return

    with open(state_path) as f:
        state = json.load(f)

    log_history = state["log_history"]

    train_loss, eval_loss, eval_per = [], [], []
    train_steps, eval_epochs = [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_loss.append(entry["loss"])
            train_steps.append(entry["step"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_per.append(entry.get("eval_PER", float("nan")))
            eval_epochs.append(entry.get("epoch", len(eval_loss)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_steps, train_loss, color="steelblue", linewidth=1.2)
    axes[0].set_title("Training loss (CTC)")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")

    axes[1].plot(eval_epochs, eval_loss, marker="o", color="darkorange")
    axes[1].set_title("Validation loss per epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("CTC Loss")

    axes[2].plot(eval_epochs, eval_per, marker="o", color="mediumseagreen")
    axes[2].set_title("Validation PER per epoch")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("PER")

    best_epoch = int(state.get("best_model_checkpoint", "epoch_0").split("-")[-1]) \
        if "best_model_checkpoint" in state else None
    if best_epoch and len(eval_per) > 0:
        axes[2].axvline(eval_epochs[np.argmin(eval_per)], color="red",
                        linestyle="--", linewidth=0.8, label="Best checkpoint")
        axes[2].legend()

    plt.tight_layout()
    out = os.path.join(DIAG_DIR, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] Training curves → {out}")
    print(f"     Best val PER : {min(eval_per):.4f}  (epoch {eval_epochs[np.argmin(eval_per)]})")
    print(f"     Final val PER: {eval_per[-1]:.4f}")


# ══════════════════════════════════════════════
# 2. LOAD MODEL + PROCESSOR
# ══════════════════════════════════════════════
def load_model_and_processor():
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=VOCAB_NAME,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=""
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        sampling_rate=16000,
        return_attention_mask=False
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # Import the custom model class from your training script
    # If running standalone, paste the class definition here instead
    sys.path.insert(0, ".")
    

    model = Wav2Vec2ForCTCWithVAD.from_pretrained(OUTPUT_DIR)
    model.eval().to(DEVICE)

    sil_id  = tokenizer.convert_tokens_to_ids("SIL")
    blank_id = tokenizer.pad_token_id
    print(f"[OK] Model loaded from {OUTPUT_DIR}")
    print(f"     Vocab size : {len(tokenizer)}")
    print(f"     blank_id   : {blank_id}  ({tokenizer.convert_ids_to_tokens(blank_id)})")
    print(f"     sil_id     : {sil_id}")
    return model, processor, tokenizer, sil_id, blank_id


# ══════════════════════════════════════════════
# 3. VAD BIAS EFFECTIVENESS
# ══════════════════════════════════════════════
def compute_energy_vad(audio, sr=16000, frame_ms=20, threshold=VAD_THRESHOLD):
    frame_size = int(sr * frame_ms / 1000)
    vad = []
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i+frame_size]
        if len(frame) == 0:
            continue
        vad.append(0.0 if np.mean(np.abs(frame)) < threshold else 1.0)
    return np.array(vad, dtype=np.float32)


def analyze_vad_bias(model, processor, dataset, tokenizer, sil_id, blank_id, n=N_SAMPLES):
    """
    For each frame, compare P(SIL) with vs without VAD bias,
    split by VAD label (0 = silence, 1 = speech).
    """
    sil_probs_silence_novad, sil_probs_speech_novad   = [], []
    sil_probs_silence_vad,   sil_probs_speech_vad     = [], []

    subset = dataset["test"].select(range(min(n, len(dataset["test"]))))

    for sample in subset:
        audio = np.array(sample["input_values"], dtype=np.float32)
        vad   = np.array(sample["vad"],          dtype=np.float32)

        iv  = torch.tensor(audio).unsqueeze(0).to(DEVICE)
        vt  = torch.tensor(vad).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Without VAD bias
            out_novad = model(input_values=iv, vad=None)
            lp_novad  = F.softmax(out_novad["logits"], dim=-1)[0].cpu().numpy()

            # With VAD bias
            out_vad   = model(input_values=iv, vad=vt)
            lp_vad    = F.softmax(out_vad["logits"],   dim=-1)[0].cpu().numpy()

        # Align VAD to logit time axis via nearest-neighbour (same as model does)
        T = lp_novad.shape[0]
        vad_aligned = np.round(
            np.interp(np.arange(T), np.linspace(0, T-1, len(vad)), vad)
        ).astype(int)

        for t in range(T):
            if vad_aligned[t] == 0:   # silence frame
                sil_probs_silence_novad.append(lp_novad[t, sil_id])
                sil_probs_silence_vad  .append(lp_vad  [t, sil_id])
            else:                      # speech frame
                sil_probs_speech_novad .append(lp_novad[t, sil_id])
                sil_probs_speech_vad   .append(lp_vad  [t, sil_id])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 50)

    axes[0].hist(sil_probs_silence_novad, bins=bins, alpha=0.6, label="No VAD", color="gray")
    axes[0].hist(sil_probs_silence_vad,   bins=bins, alpha=0.6, label="With VAD", color="steelblue")
    axes[0].set_title("P(SIL) in silence frames")
    axes[0].set_xlabel("P(SIL)"); axes[0].legend()

    axes[1].hist(sil_probs_speech_novad,  bins=bins, alpha=0.6, label="No VAD", color="gray")
    axes[1].hist(sil_probs_speech_vad,    bins=bins, alpha=0.6, label="With VAD", color="darkorange")
    axes[1].set_title("P(SIL) in speech frames")
    axes[1].set_xlabel("P(SIL)"); axes[1].legend()

    plt.tight_layout()
    out = os.path.join(DIAG_DIR, "vad_bias_effect.png")
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"\n[OK] VAD bias analysis → {out}")
    print(f"     Silence frames — mean P(SIL)  no-VAD: {np.mean(sil_probs_silence_novad):.4f}  "
          f"with-VAD: {np.mean(sil_probs_silence_vad):.4f}")
    print(f"     Speech  frames — mean P(SIL)  no-VAD: {np.mean(sil_probs_speech_novad ):.4f}  "
          f"with-VAD: {np.mean(sil_probs_speech_vad  ):.4f}")

    # Discrimination: ideal = high in silence, low in speech
    disc_novad = np.mean(sil_probs_silence_novad) - np.mean(sil_probs_speech_novad)
    disc_vad   = np.mean(sil_probs_silence_vad)   - np.mean(sil_probs_speech_vad)
    print(f"     SIL discrimination (silence-speech gap)  no-VAD: {disc_novad:+.4f}  "
          f"with-VAD: {disc_vad:+.4f}")


# ══════════════════════════════════════════════
# 4. TOKEN DISTRIBUTION IN GREEDY OUTPUT
# ══════════════════════════════════════════════
def analyze_token_distribution(model, processor, tokenizer, dataset,
                               sil_id, blank_id, n=N_SAMPLES):
    """
    In raw frame-level greedy output (before CTC collapse):
    what fraction of frames are blank, SIL, or a phoneme?
    """
    counts = defaultdict(int)
    total_frames = 0

    subset = dataset["test"].select(range(min(n, len(dataset["test"]))))

    for sample in subset:
        audio = np.array(sample["input_values"], dtype=np.float32)
        vad   = np.array(sample["vad"],          dtype=np.float32)
        iv    = torch.tensor(audio).unsqueeze(0).to(DEVICE)
        vt    = torch.tensor(vad  ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out   = model(input_values=iv, vad=vt)
            ids   = out["logits"].argmax(-1)[0].cpu().tolist()

        for id_ in ids:
            if   id_ == blank_id: counts["blank"] += 1
            elif id_ == sil_id:   counts["SIL"]   += 1
            else:                 counts["phoneme"] += 1
        total_frames += len(ids)

    print(f"\n[OK] Token distribution over {total_frames} frames ({n} samples, with VAD):")
    for k in ["blank", "SIL", "phoneme"]:
        pct = 100 * counts[k] / total_frames
        print(f"     {k:10s}: {counts[k]:8d}  ({pct:.1f}%)")

    # Pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [f"blank\n{100*counts['blank']/total_frames:.1f}%",
              f"SIL\n{100*counts['SIL']/total_frames:.1f}%",
              f"phoneme\n{100*counts['phoneme']/total_frames:.1f}%"]
    ax.pie([counts["blank"], counts["SIL"], counts["phoneme"]],
           labels=labels, colors=["#9ecae1", "#fc8d59", "#74c476"])
    ax.set_title("Frame-level token distribution (greedy, with VAD)")
    out = os.path.join(DIAG_DIR, "token_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     Chart → {out}")

    if counts["SIL"] == 0:
        print("\n  *** WARNING: SIL token NEVER predicted. "
              "CTC gradient dominated VAD bias. "
              "Try reducing alpha or applying VAD only at inference. ***")
    elif counts["SIL"] / total_frames > 0.3:
        print("\n  *** WARNING: SIL is over 30% of frames — alpha may be too high. ***")


# ══════════════════════════════════════════════
# 5. PER WITH vs WITHOUT VAD BIAS
# ══════════════════════════════════════════════
def ctc_collapse_ids(ids, blank_id):
    out, prev = [], None
    for i in ids:
        if i != blank_id and i != prev:
            out.append(i)
        prev = i
    return out


def evaluate_per(model, processor, tokenizer, dataset, sil_id, blank_id,
                 use_vad=True, n=N_SAMPLES):
    pers = []
    subset = dataset["test"].select(range(min(n, len(dataset["test"]))))

    for sample in subset:
        audio  = np.array(sample["input_values"], dtype=np.float32)
        vad    = np.array(sample["vad"],          dtype=np.float32)
        labels = np.array(sample["labels"])

        iv = torch.tensor(audio).unsqueeze(0).to(DEVICE)
        vt = torch.tensor(vad  ).unsqueeze(0).to(DEVICE) if use_vad else None

        with torch.no_grad():
            out  = model(input_values=iv, vad=vt)
            ids  = out["logits"].argmax(-1)[0].cpu().tolist()

        pred = ctc_collapse_ids(ids, blank_id)
        pred = [i for i in pred if i != sil_id]   # exclude SIL from PER

        ref  = [int(i) for i in labels if i not in (-100, blank_id)]

        pred_toks = [tokenizer.convert_ids_to_tokens(i) for i in pred]
        ref_toks  = [tokenizer.convert_ids_to_tokens(i) for i in ref]

        pers.append(editdistance.eval(pred_toks, ref_toks) / max(1, len(ref_toks)))

    return np.mean(pers), np.std(pers)


def compare_per(model, processor, tokenizer, dataset, sil_id, blank_id, n=N_SAMPLES):
    print(f"\n[...] Computing PER (n={n}) — this may take a minute...")

    mean_vad,  std_vad  = evaluate_per(model, processor, tokenizer, dataset,
                                        sil_id, blank_id, use_vad=True,  n=n)
    mean_novad, std_novad = evaluate_per(model, processor, tokenizer, dataset,
                                          sil_id, blank_id, use_vad=False, n=n)

    print(f"\n[OK] Test PER comparison:")
    print(f"     With VAD bias : {mean_vad:.4f} ± {std_vad:.4f}")
    print(f"     Without VAD   : {mean_novad:.4f} ± {std_novad:.4f}")
    delta = mean_novad - mean_vad
    print(f"     Δ PER         : {delta:+.4f}  "
          f"({'VAD helps' if delta > 0 else 'VAD hurts or neutral'})")


# ══════════════════════════════════════════════
# 6. PER-PHONEME ERROR RATE
# ══════════════════════════════════════════════
def per_phoneme_analysis(model, processor, tokenizer, dataset,
                          sil_id, blank_id, n=N_SAMPLES):
    """
    Substitution / deletion / insertion counts per phoneme in the reference.
    Uses a simple sequence alignment (editdistance per phoneme not available directly,
    so we use the Levenshtein alignment).
    """
    from difflib import SequenceMatcher

    sub_counts  = defaultdict(int)
    del_counts  = defaultdict(int)
    ref_totals  = defaultdict(int)

    subset = dataset["test"].select(range(min(n, len(dataset["test"]))))

    for sample in subset:
        audio  = np.array(sample["input_values"], dtype=np.float32)
        vad    = np.array(sample["vad"],          dtype=np.float32)
        labels = np.array(sample["labels"])

        iv = torch.tensor(audio).unsqueeze(0).to(DEVICE)
        vt = torch.tensor(vad  ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(input_values=iv, vad=vt)
            ids = out["logits"].argmax(-1)[0].cpu().tolist()

        pred = ctc_collapse_ids(ids, blank_id)
        pred = [tokenizer.convert_ids_to_tokens(i) for i in pred if i != sil_id]
        ref  = [tokenizer.convert_ids_to_tokens(int(i))
                for i in labels if i not in (-100, blank_id)]

        for tok in ref:
            ref_totals[tok] += 1

        sm = SequenceMatcher(None, ref, pred)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace":
                for tok in ref[i1:i2]:
                    sub_counts[tok] += 1
            elif tag == "delete":
                for tok in ref[i1:i2]:
                    del_counts[tok] += 1

    # Build error rate per phoneme
    phonemes = sorted(ref_totals.keys(), key=lambda p: -ref_totals[p])
    top = phonemes[:30]

    error_rates = [(p, (sub_counts[p] + del_counts[p]) / ref_totals[p]) for p in top]
    error_rates.sort(key=lambda x: -x[1])

    print("\n[OK] Per-phoneme error rate (top 30 by frequency, sorted by error rate):")
    print(f"     {'Phoneme':10s}  {'Error rate':>10s}  {'Count':>8s}")
    for p, er in error_rates:
        bar = "█" * int(er * 20)
        print(f"     {p:10s}  {er:.3f}  {bar:20s}  n={ref_totals[p]}")

    # Bar chart
    labels_plot = [p for p, _ in error_rates]
    values      = [er for _, er in error_rates]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(labels_plot, values, color="steelblue")
    ax.set_xlabel("Phoneme"); ax.set_ylabel("Error rate (sub+del)")
    ax.set_title("Per-phoneme error rate (with VAD, test set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(DIAG_DIR, "per_phoneme_error.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"     Chart → {out}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Pre-inference model diagnostics")
    print("=" * 60)

    # 1. Training curves (no model needed)
    plot_training_curves()

    # 2. Load model
    model, processor, tokenizer, sil_id, blank_id = load_model_and_processor()

    # 3. Load dataset
    print(f"\n[...] Loading preprocessed dataset from {PREPROC_PATH}")
    dataset = load_from_disk(PREPROC_PATH)
    print(f"[OK]  Splits: { {k: len(v) for k, v in dataset.items()} }")

    # 4. VAD bias analysis
    analyze_vad_bias(model, processor, dataset, tokenizer, sil_id, blank_id)

    # 5. Token distribution
    analyze_token_distribution(model, processor, tokenizer, dataset,
                                sil_id, blank_id)

    # 6. PER with vs without VAD
    compare_per(model, processor, tokenizer, dataset, sil_id, blank_id)

    # 7. Per-phoneme breakdown
    per_phoneme_analysis(model, processor, tokenizer, dataset, sil_id, blank_id)

    print(f"\n{'=' * 60}")
    print(f"  All diagnostics saved to: {DIAG_DIR}/")
    print(f"{'=' * 60}")