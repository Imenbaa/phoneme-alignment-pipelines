import os
import json
import argparse
import numpy as np
import torch
import editdistance
from datasets import load_from_disk, load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

def ctc_collapse(token_ids: list, blank_id: int, pad_id: int) -> list:
    """
    Remove consecutive duplicate tokens and blank/pad tokens.
    Replicates CTC greedy decoding collapse.
    """
    collapsed = []
    prev = None
    for t in token_ids:
        if t != prev:
            if t not in (blank_id, pad_id):
                collapsed.append(t)
        prev = t
    return collapsed
 
 
def compute_per(pred_ids: list, ref_ids: list) -> float:
    """Edit distance PER, returns value in [0, inf)."""
    if len(ref_ids) == 0:
        return 0.0 if len(pred_ids) == 0 else 1.0
    return editdistance.eval(pred_ids, ref_ids) / len(ref_ids)


def evaluate(args):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # ── Load processor ────────────────────────────────────────────────────────
    print("Loading processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file           = args.vocab_file,
        unk_token            = "[UNK]",
        pad_token            = "[PAD]",
        word_delimiter_token = "",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size          = 1,
        sampling_rate         = 16000,
        padding_value         = 0.0,
        do_normalize          = True,
        return_attention_mask = True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor = feature_extractor,
        tokenizer         = tokenizer,
    )
 
    blank_id = 0
    pad_id   = processor.tokenizer.pad_token_id
 
    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model_path}...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_path,
        ctc_loss_reduction = "mean",
        ctc_zero_infinity  = True,
        pad_token_id       = pad_id,
        vocab_size         = len(tokenizer),
    ).to(device)
    model.eval()
 
    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset...")
    dataset = load_dataset("audiofolder", data_dir=args.data_dir)
    data    = dataset.cast_column("audio", Audio(sampling_rate=16000))
    mode    = "audiofolder"
    print("  Loaded as audiofolder dataset.")
 
    print(f"  {len(data)} samples ")
 
    # ── Evaluate sample by sample ─────────────────────────────────────────────
    all_per  = []
    all_pred = []
    all_ref  = []
    errors   = []
 
    print("\nEvaluating...")
    for i, sample in enumerate(data):

        audio_array = sample["audio"]["array"].astype(np.float32)
        inputs = feature_extractor(
            audio_array,
            sampling_rate         = 16000,
            return_tensors        = "pt",
            padding               = False,
            return_attention_mask = True,
        )
        input_values   = inputs.input_values.to(device)
 
                col = args.phoneme_column
                if col not in sample:
                    raise KeyError(
                        f"Column '{col}' not found. "
                        f"Available columns: {list(sample.keys())}"
                    )
                ref_ids = tokenizer(sample[col]).input_ids
 
            # ── Forward pass ──────────────────────────────────────────────────
            with torch.no_grad():
                logits = model(
                    input_values,
                    attention_mask=attention_mask,
                ).logits                              # (1, T, vocab)
 
            # ── Greedy CTC decode ─────────────────────────────────────────────
            pred_ids_raw = logits[0].argmax(dim=-1).cpu().tolist()
            pred_ids     = ctc_collapse(pred_ids_raw, blank_id, pad_id)
 
            pred_tokens  = tokenizer.convert_ids_to_tokens(pred_ids)
            ref_tokens   = tokenizer.convert_ids_to_tokens(ref_ids)
 
            # ── PER ───────────────────────────────────────────────────────────
            per = compute_per(pred_ids, ref_ids)
            all_per.append(per)
            all_pred.append(pred_tokens)
            all_ref.append(ref_tokens)
 
            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"  [{i+1:5d}/{len(data)}]  "
                    f"running PER: {np.mean(all_per)*100:.2f}%  "
                    f"this sample: {per*100:.2f}%"
                )
 
    # ── Aggregate ─────────────────────────────────────────────────────────────
    final_per  = float(np.mean(all_per))   if all_per else float("nan")
    median_per = float(np.median(all_per)) if all_per else float("nan")
    worst_idx  = int(np.argmax(all_per))   if all_per else -1
    best_idx   = int(np.argmin(all_per))   if all_per else -1
 
    print("\n" + "=" * 60)
    print(f"  Samples evaluated : {len(all_per)} / {len(data)}")
    print(f"  Samples failed    : {len(errors)}")
    print(f"  Mean PER          : {final_per  * 100:.2f}%")
    print(f"  Median PER        : {median_per * 100:.2f}%")
    if worst_idx >= 0:
        print(f"  Worst PER         : {all_per[worst_idx]*100:.2f}%  (sample {worst_idx})")
        print(f"    ref : {' '.join(all_ref[worst_idx])}")
        print(f"    pred: {' '.join(all_pred[worst_idx])}")
    if best_idx >= 0:
        print(f"  Best PER          : {all_per[best_idx]*100:.2f}%  (sample {best_idx})")
    print("=" * 60)
 
    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "model_path"     : args.model_path,
        "split"          : args.split,
        "n_samples"      : len(all_per),
        "n_failed"       : len(errors),
        "mean_per"       : final_per,
        "mean_per_pct"   : round(final_per  * 100, 4),
        "median_per"     : median_per,
        "median_per_pct" : round(median_per * 100, 4),
        "per_per_sample" : all_per,
        "predictions"    : all_pred,
        "references"     : all_ref,
        "errors"         : errors,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")
 
    return final_per
