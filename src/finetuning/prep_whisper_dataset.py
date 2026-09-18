#!/usr/bin/env python
"""
Pre-compute Whisper mel features + tokenized labels for a CSV-based dataset,
chunk by chunk, with auto-resume. Run this ONCE before training, with ONE
process (no torchrun).

Why chunks: prepare on the full ~430k CV-fr corpus has been dying silently after
~30-50 min on this server. Processing 50k samples at a time keeps each chunk
under ~8 minutes, well below the failure threshold. If a chunk dies, finished
chunks stay on disk; re-running the script picks up at the next missing chunk.

Output layout:
    <output_dir>/train_chunks/chunk_00000/   # Arrow shard for samples 0..49999
    <output_dir>/train_chunks/chunk_00001/   # Arrow shard for samples 50000..99999
    ...
    <output_dir>/eval/                       # eval split, single shard

Training (`train_whisper.py --prepared_dataset <output_dir>`) automatically
concatenates the train chunks back into one dataset.

Example:
    python prep_whisper_dataset.py \
        --train_csv /vol/.../commonvoice_fr_wavlm.csv \
        --eval_csv  /vol/.../commonvoice_fr_wavlm_dev.csv \
        --clips_dir /vol/corpora/CommonVoice/cv-corpus-19.0-2024-09-13/fr/clips \
        --vocab_file /vol/.../vocab_CV.json \
        --model_path /vol/.../whisper-large-v3 \
        --output_dir /vol/experiments/cache_imbenamor/whisper_prepared \
        --chunk_size 50000
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperFeatureExtractor, Wav2Vec2PhonemeCTCTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--clips_dir", required=True)
    p.add_argument("--vocab_file", required=True)
    p.add_argument("--model_path", required=True,
                   help="Whisper checkpoint dir (used only for its WhisperFeatureExtractor)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--path_column", default="path")
    p.add_argument("--phonemes_column", default="phonemes")
    p.add_argument("--chunk_size", type=int, default=50000,
                   help="rows per train chunk. Smaller = safer (more chunks) but more "
                        "overhead. 50k = ~8 min per chunk on this server.")
    p.add_argument("--max_duration_s", type=float, default=20.0)
    p.add_argument("--min_duration_s", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)
    return p.parse_args()


def load_csv_split(csv_path, clips_dir, path_column, phonemes_column):
    df = pd.read_csv(csv_path, na_filter=False)
    for col in (path_column, phonemes_column):
        if col not in df.columns:
            raise ValueError(f"'{col}' not in {csv_path}; columns: {list(df.columns)}")
    df = df[df[phonemes_column].str.strip().astype(bool)]
    df = df.assign(audio=df[path_column].apply(lambda p: os.path.join(clips_dir, p)))
    ds = Dataset.from_pandas(df[["audio", phonemes_column]].reset_index(drop=True))
    if phonemes_column != "phonemes":
        ds = ds.rename_column(phonemes_column, "phonemes")
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_chunks_dir = os.path.join(args.output_dir, "train_chunks")
    eval_out = os.path.join(args.output_dir, "eval")
    os.makedirs(train_chunks_dir, exist_ok=True)

    # ---- tokenizer + feature extractor ----------------------------------- #
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
        args.vocab_file, unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token=None, phone_delimiter_token=" ",
        do_phonemize=False,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)
    print(f"[vocab] {args.vocab_file} ({len(tokenizer)} tokens)")
    print(f"[feature_extractor] {args.model_path} "
          f"(n_mels={feature_extractor.feature_size})")

    min_len = int(args.min_duration_s * 16000)
    max_len = int(args.max_duration_s * 16000)

    def prepare(batch):
        audio = batch["audio"]
        feats = feature_extractor(audio["array"], sampling_rate=16000)
        batch["input_features"] = feats.input_features[0].astype(np.float16)
        batch["input_length"] = len(audio["array"])
        batch["labels"] = tokenizer(batch["phonemes"], do_phonemize=False).input_ids
        return batch

    # ---- eval split (one shard, prepare only if missing) ----------------- #
    if os.path.isdir(eval_out) and os.listdir(eval_out):
        print(f"[eval] already prepared at {eval_out}, skipping")
    else:
        eval_ds = load_csv_split(args.eval_csv, args.clips_dir,
                                 args.path_column, args.phonemes_column)
        if args.max_eval_samples:
            eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
        print(f"[eval] preparing {len(eval_ds)} samples ...")
        eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names,
                              num_proc=1, desc="prepare-eval")
        eval_ds.save_to_disk(eval_out)
        print(f"[eval] saved -> {eval_out}")
        del eval_ds

    # ---- train split: chunked with resume -------------------------------- #
    train_ds = load_csv_split(args.train_csv, args.clips_dir,
                              args.path_column, args.phonemes_column)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    n_train = len(train_ds)
    n_chunks = (n_train + args.chunk_size - 1) // args.chunk_size
    print(f"[train] total={n_train}  chunk_size={args.chunk_size}  n_chunks={n_chunks}")

    done = set()
    for d in os.listdir(train_chunks_dir):
        path = os.path.join(train_chunks_dir, d)
        if d.startswith("chunk_") and os.path.isdir(path) and os.listdir(path):
            try:
                done.add(int(d.split("_")[1]))
            except ValueError:
                pass
    if done:
        print(f"[train] already done chunks: {sorted(done)}")

    for chunk_idx in range(n_chunks):
        if chunk_idx in done:
            continue
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, n_train)
        print(f"[train chunk {chunk_idx}/{n_chunks-1}] samples [{start}, {end})")
        chunk = train_ds.select(range(start, end))
        chunk = chunk.map(prepare, remove_columns=chunk.column_names,
                          num_proc=1, desc=f"prepare-train-chunk-{chunk_idx}")
        chunk = chunk.filter(lambda l: min_len < l < max_len,
                             input_columns=["input_length"])
        chunk_dir = os.path.join(train_chunks_dir, f"chunk_{chunk_idx:05d}")
        chunk.save_to_disk(chunk_dir)
        print(f"[train chunk {chunk_idx}] saved {len(chunk)} samples -> {chunk_dir}")
        del chunk

    print(f"[done] all chunks complete. Use:")
    print(f"    --prepared_dataset {args.output_dir}")


if __name__ == "__main__":
    main()
