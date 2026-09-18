import os
import json
import argparse
import torch
import torchaudio
import numpy as np
import editdistance
from pathlib import Path
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
import tgt
import csv
from pathlib import Path
from collections import defaultdict
from VAD_chunk import *


sampa_to_api_single = {'a': 'a', 'e': 'e','i': 'i','o': 'o','u': 'u','y': 'y','2': 'ø','9': '9','@': 'ə','E': 'ɛ','O': 'ɔ','a~': '@','e~': '5','9~': '1','o~': '§','b': 'b','d': 'd','f': 'f', 'g': 'ɡ', 'k': 'k', 'l': 'l','m': 'm','n': 'n', 'n=':'n', 'p': 'p','s': 's','t': 't','v': 'v','w': 'w','z': 'z','j': 'j', 'R': 'ʁ','N': 'ŋ', 'H': 'ɥ','J': 'ɲ','S': 'ʃ', 'Z=': 'dʒ','s': 'ts','Z': 'dʒ','m=': 'm','_': '_','spn': 'spn','unk': 'spn',"%":'spn',"?":"spn","0":"spn"}
model_path = "results/w2vCTC/checkpoint-23430"
vocab_path = "vocab_w2vCTC.json"
wav_dir = "/vol/corpora/Rhapsodie/wav16k_corrected"
tg_dir = "/vol/corpora/Rhapsodie/TextGrids-fev2013/" 
tg_tier ="phone"
output = "results/per_evaluation.txt"
style_csv = "/vol/corpora/Rhapsodie/wav_style.csv"
def map_ref_to_api_single(ph):
     
    if ph in sampa_to_api_single:
        mapped = sampa_to_api_single[ph]
        return mapped
    else:
        if ph!= None and ph != "":
            return ph
def read_textgrid(tg_path, tier_name="phone"):

    tg        = tgt.io.read_textgrid(tg_path)
    tier      = tg.get_tier_by_name(tier_name)
    silence   = {"", "SIL", "sil", "spn", "SP", "<SIL>", "_","0","fe~","sjo~","Ra~"}
    phonemes = []
    for iv in tier.intervals:
        label = iv.text.strip()
        phoneme=map_ref_to_api_single(label)
        if phoneme in silence:
            continue
        phonemes.append(phoneme)
    return phonemes



def ctc_collapse(token_ids: list, blank_id: int) -> list:
    """
    Standard CTC greedy decode:
    remove consecutive duplicates then remove blank tokens.
    """
    collapsed, prev = [], None
    for t in token_ids:
        if t != prev:
            if t != blank_id:
                collapsed.append(t)
        prev = t
    return collapsed
def evaluate_file(
    wav_path:  str,
    tg_path:   str,
    tier_name: str,
    model:     Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device:    torch.device,
) -> dict:
    """
    Evaluate one (wav, textgrid) pair using VAD chunking.
    Each chunk is decoded independently, then phoneme sequences
    are concatenated and compared against the full reference.
    """
    # ── Load and preprocess audio ────────────────────────────────────
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)          # stereo → mono
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
 
    wav      = torch.from_numpy(audio.astype(np.float32))
    blank_id = processor.tokenizer.pad_token_id  # [PAD] = 0 = CTC blank
 
    # ── VAD chunking ─────────────────────────────────────────────────
    chunks = vad_chunk_with_timestamps(wav,max_chunk_duration=30.0,max_pause_duration=0.6)
 
    if not chunks:
        # Fallback: whole file as one chunk
        chunks = [{"start": 0.0, "end": wav.shape[0] / 16000}]
 
    # ── Decode each chunk, concatenate phonemes ───────────────────────
    hyp = []
    for chunk in chunks:
        start_sample = int(chunk["start"] * 16000)
        end_sample   = int(chunk["end"]   * 16000)
        chunk_audio  = wav[start_sample:end_sample]
 
        # Skip chunks that are too short for the model
        if chunk_audio.shape[0] < 400:   # < 25ms
            continue
 
        inputs = processor(
            chunk_audio.numpy(),
            sampling_rate  = 16000,
            return_tensors = "pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
 
        with torch.no_grad():
            logits = model(**inputs).logits   # (1, T, vocab)
 
        pred_ids  = logits.argmax(dim=-1)[0].tolist()
        collapsed = ctc_collapse(pred_ids, blank_id)
        chunk_hyp = [
            processor.tokenizer.convert_ids_to_tokens(i)
            for i in collapsed
        ]
        hyp.extend(chunk_hyp)
 
    # ── Reference from TextGrid ──────────────────────────────────────
    ref = read_textgrid(tg_path, tier_name)
 
    # ── PER ──────────────────────────────────────────────────────────
    n_errors = editdistance.eval(hyp, ref)
    n_ref    = max(len(ref), 1)
    per      = n_errors / n_ref
 
    return {
        "file":     os.path.basename(wav_path),
        "hyp":      hyp,
        "ref":      ref,
        "per":      per,
        "n_ref":    len(ref),
        "n_errors": n_errors,
        "n_chunks": len(chunks),
    }
def load_style_map(csv_path: str, file_col: str, style_col: str) -> dict:
    """
    Load a CSV mapping filename → style.
    Returns dict: {stem: style}
    Handles filenames with or without .wav extension.
    """
    style_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if file_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{file_col}' not found in {csv_path}.\n"
                f"Available columns: {reader.fieldnames}"
            )
        if style_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{style_col}' not found in {csv_path}.\n"
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            fname = row[file_col].strip()
            # normalize to stem (no extension)
            stem  = Path(fname).stem
            style = row[style_col].strip()
            style_map[stem] = style
    return style_map
 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Processor ────────────────────────────────────────────────────────────
print(f"Loading processor from {vocab_path} ...")
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file           = vocab_path,
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

# ── Model ─────────────────────────────────────────────────────────────────
print(f"Loading model from {model_path} ...")
model = Wav2Vec2ForCTC.from_pretrained(
    model_path,
    ctc_loss_reduction = "mean",
    ctc_zero_infinity  = True,
    pad_token_id       = processor.tokenizer.pad_token_id,
    vocab_size         = len(processor.tokenizer),
).to(device)
model.eval()
print(f"Vocab size: {len(processor.tokenizer)}")
# ── Style map ─────────────────────────────────────────────────────────────
style_map = {}
print(f"Loading style map from {style_csv} ...")
style_map = load_style_map(style_csv, "file", "style")
styles    = set(style_map.values())
print(f"Found {len(styles)} styles: {sorted(styles)}")
print(f"Style map covers {len(style_map)} files\n")

# ── Collect file pairs ────────────────────────────────────────────────────
wav_dir   = Path(wav_dir)
tg_dir    = Path(tg_dir)
wav_files = sorted(wav_dir.glob("*.wav"))

pairs = []
for wav_path in wav_files:
    tg_path = tg_dir / (wav_path.stem + "-Pro.TextGrid")

    if tg_path.exists():
        pairs.append((str(wav_path), str(tg_path)))
print(f"Evaluating {len(pairs)} file pairs ...\n")

style_map1={}
for i in style_map.keys():
    style_map1["Rhap-"+i]=style_map[i]
# ── Evaluate file by file ─────────────────────────────────────────────────
results      = []
total_errors = 0
total_ref    = 0
failed       = []

for i, (wav_path, tg_path) in enumerate(pairs):
    r = evaluate_file(
        wav_path, tg_path, tg_tier,
        model, processor, device,
    )
    # Attach style if available
    stem       = Path(wav_path).stem
    r["style"] = style_map1.get(stem, "unknown")

    results.append(r)
    total_errors += r["n_errors"]
    total_ref    += r["n_ref"]

    if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
        running_per = total_errors / max(total_ref, 1)
        print(
            f"  [{i+1:4d}/{len(pairs)}]  "
            f"running PER = {running_per*100:.2f}%"
        )




# ── Per-style summary ─────────────────────────────────────────────────────
style_stats = None
if style_map:
    style_errors = defaultdict(int)
    style_ref    = defaultdict(int)
    style_files  = defaultdict(int)

    for r in results:
        s = r["style"]
        style_errors[s] += r["n_errors"]
        style_ref[s]    += r["n_ref"]
        style_files[s]  += 1

    style_stats = {}
    for s in sorted(style_errors.keys()):
        style_stats[s] = {
            "per":   style_errors[s] / max(style_ref[s], 1),
            "files": style_files[s],
            "refs":  style_ref[s],
            "errors": style_errors[s],
        }

    print(f"\n{'='*55}")
    print(f"  PER BY STYLE")
    print(f"{'='*55}")
    for s, v in style_stats.items():
        print(
            f"  {s:20s}  PER={v['per']*100:5.2f}%  "
            f"files={v['files']:4d}  phones={v['refs']:6d}"
        )
    print(f"{'='*55}\n")
# ── Summary ───────────────────────────────────────────────────────────────
global_per = total_errors / max(total_ref, 1)
mean_per   = float(np.mean([r["per"] for r in results])) if results else 0.0

print(f"\n{'='*55}")
print(f"  Files evaluated : {len(results)}")
print(f"  Files failed    : {len(failed)}")
print(f"  Total phonemes  : {total_ref}")
print(f"  Total errors    : {total_errors}")
print(f"  Global PER      : {global_per*100:.2f}%")
print(f"  Mean file PER   : {mean_per*100:.2f}%")
print(f"{'='*55}\n")

# ── Save detailed output ──────────────────────────────────────────────────
if output:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        f.write(f"Global PER : {global_per*100:.2f}%\n")
        f.write(f"Mean PER   : {mean_per*100:.2f}%\n")
        f.write(f"Files      : {len(results)}\n")
        f.write(f"Failed     : {len(failed)}\n")
        f.write("="*55 + "\n\n")

        for r in sorted(results, key=lambda x: -x["per"]):
            f.write(f"File : {r['file']}\n")
            f.write(
                f"PER  : {r['per']*100:.2f}%  "
                f"({r['n_errors']} errors / {r['n_ref']} phones)\n"
            )
            f.write(f"REF  : {' '.join(r['ref'])}\n")
            f.write(f"HYP  : {' '.join(r['hyp'])}\n\n")

        if failed:
            f.write("="*55 + "\nFAILED\n" + "="*55 + "\n")
            for fname, err in failed:
                f.write(f"{fname}: {err}\n")

        if style_stats:
            f.write("\n" + "="*55 + "\nPER BY STYLE\n" + "="*55 + "\n")
            for s, v in style_stats.items():
                f.write(
                    f"{s:20s}  PER={v['per']*100:.2f}%  "
                    f"files={v['files']}  phones={v['refs']}  "
                    f"errors={v['errors']}\n"
                )

    print(f"Detailed results saved to: {output}")