#!/usr/bin/env python
"""
Fine-tune WavLM for phoneme recognition (CTC) on a pre-phonemized CSV dataset.
Runs entirely on the (offline) server -- no Hub access at all.

Expects two CSVs (train + dev) with at least:
  - `path`     : audio filename, resolved against --clips_dir
  - `phonemes` : space-separated IPA tokens (already computed; no espeak at runtime)

By default the phoneme vocab is BUILT from the train `phonemes` column, so it
always matches the data. Pass --vocab_file only if you need a fixed external
vocab (e.g. shared across corpora); then it's used as-is.

Still required ON THE SERVER:
  * WavLM weights, copied locally -> pass via --model_path
  * pip deps: torch torchaudio transformers datasets jiwer soundfile librosa accelerate tensorboard
  * mp3 decoding needs a recent libsndfile (>=1.1) or ffmpeg available.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import jiwer
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    WavLMForCTC,
    TrainingArguments,
    Trainer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/commonvoice_fr_wavlm_train.csv")
    p.add_argument("--eval_csv", default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/commonvoice_fr_wavlm_dev.csv")
    p.add_argument("--clips_dir", default="/vol/corpora/CommonVoice/cv-corpus-19.0-2024-09-13/fr/clips")
    p.add_argument("--vocab_file", default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/vocab_CV.json",
                   help="OPTIONAL fixed vocab JSON; if omitted, vocab is built from the phonemes column")
    p.add_argument("--path_column", default="path")
    p.add_argument("--phonemes_column", default="phonemes")

    p.add_argument("--model_path", default="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wavlm-large",
                   help="local directory with WavLM weights")
    p.add_argument("--output_dir", default="/vol/experiments3/imbenamor/TAPAS-FRAIS/src/finetuning/wavlm-fr-phoneme-large")

    p.add_argument("--num_train_epochs", type=float, default=15.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=1500)
    p.add_argument("--logging_steps", type=int, default=500)

    p.add_argument("--max_duration_s", type=float, default=20.0)
    p.add_argument("--min_duration_s", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--num_proc", type=int, default=1)
    return p.parse_args()


# ---- CSV loader: expects pre-phonemized rows ----------------------------- #
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


# ---- build vocab straight from the phonemes column ----------------------- #
def build_vocab(phoneme_strings, out_dir):
    phones = set()
    for s in phoneme_strings:
        phones.update(s.split())
    phones.discard("")
    vocab = {p: i for i, p in enumerate(sorted(phones))}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "vocab.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[vocab] built {len(vocab)} tokens from data -> {path}")
    return path


@dataclass
class DataCollatorCTCWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2PhonemeCTCTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- load CSVs ------------------------------------------------------- #
    train_ds = load_csv_split(args.train_csv, args.clips_dir,
                              args.path_column, args.phonemes_column)
    eval_ds = load_csv_split(args.eval_csv, args.clips_dir,
                             args.path_column, args.phonemes_column)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    print(f"[data] train={len(train_ds)}  eval={len(eval_ds)}")

    # ---- vocab: build from the phonemes column (or use a fixed file) ----- #
    if args.vocab_file:
        vocab_path = os.path.join(args.output_dir, "vocab.json")
        shutil.copy(args.vocab_file, vocab_path)
        print(f"[vocab] using fixed vocab {args.vocab_file}")
    else:
        vocab_path = build_vocab(train_ds["phonemes"], args.output_dir)

    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token=None, phone_delimiter_token=" ",
        do_phonemize=False,
    )
    # head size = highest id + 1 (robust whether vocab is built or supplied)
    ctc_vocab_size = max(tokenizer.get_vocab().values()) + 1
    print(f"[vocab] ctc_vocab_size={ctc_vocab_size} "
          f"pad_id={tokenizer.pad_token_id} unk_id={tokenizer.unk_token_id}")

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,   # base-plus = group norm; set True for wavlm-large
    )

    # ---- feature extraction + label encoding ----------------------------- #
    min_len = int(args.min_duration_s * 16000)
    max_len = int(args.max_duration_s * 16000)

    def prepare(batch):
        audio = batch["audio"]
        batch["input_values"] = feature_extractor(
            audio["array"], sampling_rate=16000).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = tokenizer(batch["phonemes"], do_phonemize=False).input_ids
        return batch

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names,
                            num_proc=args.num_proc, desc="prepare-train")
    eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names,
                          num_proc=args.num_proc, desc="prepare-eval")
    train_ds = train_ds.filter(lambda l: min_len < l < max_len,
                               input_columns=["input_length"])
    eval_ds = eval_ds.filter(lambda l: min_len < l < max_len,
                             input_columns=["input_length"])

    # ---- model (local weights) ------------------------------------------- #
    model = WavLMForCTC.from_pretrained(
        args.model_path,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=ctc_vocab_size,
    )
    # Train Transformer encoder + CTC head; freeze CNN feature extractor + projection.
    for p in model.wavlm.parameters():
        p.requires_grad = False
    for p in model.wavlm.encoder.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ---- PER via jiwer; argmax before accumulation to keep eval memory low #
    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    def compute_metrics(pred):
        pred_ids = pred.predictions            # already argmaxed
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(label_ids, group_tokens=False)
        pairs = [(r, h) for r, h in zip(label_str, pred_str) if r.strip()]
        if not pairs:
            return {"per": 1.0}
        refs, hyps = zip(*pairs)
        return {"per": jiwer.wer(list(refs), list(hyps))}

    collator = DataCollatorCTCWithPadding(feature_extractor, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        length_column_name="input_length",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=args.num_proc,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=feature_extractor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print(f"[done] model + processor saved to {args.output_dir}")


if __name__ == "__main__":
    main()