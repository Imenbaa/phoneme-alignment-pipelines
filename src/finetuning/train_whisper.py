#!/usr/bin/env python
"""
Fine-tune the Whisper ENCODER + a fresh CTC head for phoneme recognition on a
pre-phonemized CSV dataset. Mirrors train_wavlm.py — same CSV / vocab / CTC / PER
pipeline, just swaps the backbone. The Whisper decoder is discarded.

Expects two CSVs (train + dev) with at least `path` and `phonemes` columns; the
phoneme symbols must already match the supplied --vocab_file.

Whisper vs WavLM differences worth knowing:
  - Input: log-mel spectrogram, ALWAYS padded/truncated to 30 s (mel_bins, 3000).
    WhisperFeatureExtractor handles this; loaded from preprocessor_config.json so
    it picks up the right mel-bin count (80 for older Whisper, 128 for large-v3).
  - Encoder output is ALWAYS 1500 frames -> CTC sees a fixed input length.
  - No built-in WhisperForCTC, so we define a small wrapper that loads the
    pretrained encoder weights and attaches a fresh linear head sized for the
    phoneme vocab.

Still required ON THE SERVER:
  * Whisper weights -> --model_path
        on a connected box:  huggingface-cli download openai/whisper-large-v3 \
                                 --local-dir whisper-large-v3
        then copy the folder to the server.
  * The phoneme vocab JSON (e.g. vocab_CV.json) — symbol -> id, including [UNK]/[PAD].
  * pip deps (install from a local wheelhouse if there's no PyPI):
        torch torchaudio transformers datasets jiwer soundfile librosa accelerate tensorboard
  * mp3 decoding needs a recent libsndfile (>=1.1) or ffmpeg available.

Example:
    python train_whisper.py \
        --train_csv /data/cv_fr/train.csv \
        --eval_csv  /data/cv_fr/dev.csv \
        --clips_dir /data/cv_fr/clips \
        --vocab_file /data/cv_fr/vocab_CV.json \
        --model_path /models/whisper-large-v3 \
        --output_dir ./whisper-fr-phoneme \
        --num_train_epochs 10 --per_device_train_batch_size 8
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import shutil
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
from datasets import Dataset, Audio
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperModel,
    WhisperPreTrainedModel,
    Wav2Vec2PhonemeCTCTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import CausalLMOutput


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared_dataset", default=None,
                   help="folder produced by prep_whisper_dataset.py (contains train/ and eval/). "
                        "If given, skip CSV + audio decoding + feature extraction entirely.")
    p.add_argument("--train_csv", default=None,
                   help="train CSV with at least the path + phonemes columns (ignored if --prepared_dataset is set)")
    p.add_argument("--eval_csv", default=None,
                   help="dev/eval CSV with the same schema as --train_csv (ignored if --prepared_dataset is set)")
    p.add_argument("--clips_dir", default=None,
                   help="folder containing the audio files referenced by --path_column (ignored if --prepared_dataset is set)")
    p.add_argument("--vocab_file", required=True,
                   help="phoneme vocab JSON (symbol -> id, with [UNK]/[PAD]); e.g. vocab_CV.json")
    p.add_argument("--path_column", default="path")
    p.add_argument("--phonemes_column", default="phonemes")

    p.add_argument("--model_path", required=True,
                   help="local directory with Whisper weights (e.g. ./whisper-large-v3)")
    p.add_argument("--output_dir", default="./whisper-fr-phoneme")

    p.add_argument("--num_train_epochs", type=float, default=10.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    # Whisper is more LR-sensitive than WavLM; default lower.
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)

    # Whisper hard-caps at 30 s, so keep this <= 30.
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


# ---- model: Whisper encoder + linear CTC head ---------------------------- #
class WhisperEncoderForCTC(WhisperPreTrainedModel):
    """Whisper encoder with a fresh linear CTC head; decoder is not instantiated."""
    config_class = WhisperConfig
    main_input_name = "input_features"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.0))
        # config.vocab_size is overridden at load time to match the phoneme vocab.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.post_init()

    def freeze_conv_frontend(self):
        for p in self.encoder.conv1.parameters():
            p.requires_grad = False
        for p in self.encoder.conv2.parameters():
            p.requires_grad = False

    def forward(self, input_features=None, labels=None, attention_mask=None, **kwargs):
        encoder_out = self.encoder(input_features).last_hidden_state  # (B, 1500, d_model)
        logits = self.lm_head(self.dropout(encoder_out))               # (B, 1500, V)

        loss = None
        if labels is not None:
            # CTC expects (T, B, V) log-probs in float32.
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            input_lengths = torch.full(
                (logits.shape[0],), logits.shape[1],
                dtype=torch.long, device=logits.device,
            )
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flat_targets = labels.masked_select(labels_mask)
            # cuDNN CTC has stricter shape constraints; safer to use the native impl.
            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs, flat_targets, input_lengths, target_lengths,
                    blank=self.config.pad_token_id,
                    reduction="mean", zero_infinity=True,
                )
        return CausalLMOutput(loss=loss, logits=logits)


def load_whisper_encoder_for_ctc(model_path, vocab_size, pad_token_id):
    """Build the CTC model and load ONLY the encoder weights from disk.

    Reads the sharded safetensors index, picks out keys under `model.encoder.*`,
    and loads just those tensors. Skips the ~900M decoder params entirely so
    RAM during model load stays under ~3 GB instead of ~12 GB.
    """
    import gc
    import json as _json
    from safetensors import safe_open

    config = WhisperConfig.from_pretrained(model_path)
    config.vocab_size = vocab_size
    config.pad_token_id = pad_token_id
    model = WhisperEncoderForCTC(config)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = _json.load(f)

    # Group encoder keys by their shard file so we open each shard at most once.
    enc_prefix = "model.encoder."
    by_shard: Dict[str, List[str]] = {}
    for full_key, shard_file in index["weight_map"].items():
        if full_key.startswith(enc_prefix):
            by_shard.setdefault(shard_file, []).append(full_key)

    encoder_state: Dict[str, torch.Tensor] = {}
    for shard_file, keys in by_shard.items():
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for full_key in keys:
                local_key = full_key[len(enc_prefix):]
                encoder_state[local_key] = f.get_tensor(full_key)

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"[load] loaded {len(encoder_state)} encoder tensors from {len(by_shard)} "
          f"shard(s) (missing={len(missing)}, unexpected={len(unexpected)})")
    del encoder_state
    gc.collect()
    return model


# ---- data collator ------------------------------------------------------- #
@dataclass
class DataCollatorWhisperCTC:
    feature_extractor: WhisperFeatureExtractor
    tokenizer: Wav2Vec2PhonemeCTCTokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # Whisper input_features are already a fixed (mel_bins, 3000) shape per sample.
        # Cache is stored as fp16 to save disk; cast to fp32 here because the
        # Whisper conv frontend's weights are fp32 (autocast will redo the fp16
        # conversion inside the forward pass when fp16=True).
        input_features = torch.stack(
            [torch.as_tensor(f["input_features"]).float() for f in features]
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return {"input_features": input_features, "labels": labels}


def main():
    args = parse_args()

    # ---- tokenizer + feature extractor (always needed) ------------------- #
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(args.vocab_file, os.path.join(args.output_dir, "vocab.json"))
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
        args.vocab_file, unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token=None, phone_delimiter_token=" ",
        do_phonemize=False,
    )
    vocab_dict = tokenizer.get_vocab()
    ctc_vocab_size = max(vocab_dict.values()) + 1
    print(f"[vocab] using {args.vocab_file} "
          f"(len(tokenizer)={len(tokenizer)}, max_id={max(vocab_dict.values())}, "
          f"ctc_vocab_size={ctc_vocab_size})")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)

    # ---- load data: either prepared-on-disk OR CSV + prepare-on-the-fly -- #
    if args.prepared_dataset:
        from datasets import load_from_disk, concatenate_datasets
        eval_ds = load_from_disk(os.path.join(args.prepared_dataset, "eval"))
        # Train split lives as one chunk per directory under train_chunks/
        train_chunks_dir = os.path.join(args.prepared_dataset, "train_chunks")
        chunk_dirs = sorted(
            os.path.join(train_chunks_dir, d)
            for d in os.listdir(train_chunks_dir)
            if d.startswith("chunk_") and os.path.isdir(os.path.join(train_chunks_dir, d))
        )
        if not chunk_dirs:
            raise ValueError(f"No chunk_* subdirs found under {train_chunks_dir}")
        train_ds = concatenate_datasets([load_from_disk(d) for d in chunk_dirs])
        if args.max_train_samples:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        if args.max_eval_samples:
            eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
        print(f"[data/prepared] {args.prepared_dataset}  "
              f"train={len(train_ds)} (from {len(chunk_dirs)} chunks)  "
              f"eval={len(eval_ds)}")
    else:
        if not (args.train_csv and args.eval_csv and args.clips_dir):
            raise ValueError("Without --prepared_dataset you must provide "
                             "--train_csv, --eval_csv and --clips_dir.")
        train_ds = load_csv_split(args.train_csv, args.clips_dir,
                                  args.path_column, args.phonemes_column)
        eval_ds = load_csv_split(args.eval_csv, args.clips_dir,
                                 args.path_column, args.phonemes_column)
        if args.max_train_samples:
            train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        if args.max_eval_samples:
            eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
        print(f"[data/csv] train={len(train_ds)}  eval={len(eval_ds)}")

        min_len = int(args.min_duration_s * 16000)
        max_len = int(args.max_duration_s * 16000)

        def prepare(batch):
            audio = batch["audio"]
            feats = feature_extractor(audio["array"], sampling_rate=16000)
            batch["input_features"] = feats.input_features[0].astype(np.float16)
            batch["input_length"] = len(audio["array"])
            batch["labels"] = tokenizer(batch["phonemes"], do_phonemize=False).input_ids
            return batch

        train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names,
                                num_proc=args.num_proc, desc="prepare-train")
        eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names,
                              num_proc=args.num_proc, desc="prepare-eval")
        train_ds = train_ds.filter(lambda l: min_len < l < max_len,
                                   input_columns=["input_length"])

    # Defensive: scan actual labels and bump vocab_size to whatever the tokenizer
    # really emits. get_vocab() can miss phantom added tokens, so the only safe
    # ground truth is the labels themselves.
    observed_max = 0
    for split in (train_ds, eval_ds):
        for lab in split["labels"]:
            if lab:
                m = max(lab)
                if m > observed_max:
                    observed_max = m
    if observed_max + 1 > ctc_vocab_size:
        print(f"[vocab] bump: observed max label id={observed_max} > "
              f"current ctc_vocab_size={ctc_vocab_size}; raising to {observed_max + 1}")
        ctc_vocab_size = observed_max + 1
    else:
        print(f"[vocab] observed max label id={observed_max} "
              f"(fits ctc_vocab_size={ctc_vocab_size})")
    print(f"[vocab] tokenizer.pad_token_id={tokenizer.pad_token_id} "
          f"unk_token_id={tokenizer.unk_token_id}")

    # ---- model (local weights) ------------------------------------------- #
    model = load_whisper_encoder_for_ctc(
        args.model_path,
        vocab_size=ctc_vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    # ------------------------------------------------------------------
    # Freeze everything
    # ------------------------------------------------------------------
    for p in model.parameters():
        p.requires_grad = False
    
    # ------------------------------------------------------------------
    # Unfreeze Whisper Transformer encoder
    # ------------------------------------------------------------------
    for p in model.encoder.layers.parameters():
        p.requires_grad = True
    
    # Final encoder LayerNorm
    for p in model.encoder.layer_norm.parameters():
        p.requires_grad = True
    
    # ------------------------------------------------------------------
    # Unfreeze CTC head
    # ------------------------------------------------------------------
    for p in model.lm_head.parameters():
        p.requires_grad = True
    
    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(
        f"[freeze] trainable params: "
        f"{trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%)"
    )
    # ---- PER via jiwer --------------------------------------------------- #
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(label_ids, group_tokens=False)
        pairs = [(r, h) for r, h in zip(label_str, pred_str) if r.strip()]
        if not pairs:
            return {"per": 1.0}
        refs, hyps = zip(*pairs)
        return {"per": jiwer.wer(list(refs), list(hyps))}

    collator = DataCollatorWhisperCTC(feature_extractor, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=-1,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
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
        tokenizer=feature_extractor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    print(f"[done] model + processor saved to {args.output_dir}")


if __name__ == "__main__":
    main()
