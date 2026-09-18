import os
import json
import argparse
import logging
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict, Union

from sklearn.model_selection import train_test_split
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import evaluate
from torch.utils.data import Dataset
import torch.nn as nn

# =========================
# Logging Setup
# =========================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# =========================
# Dataset
# =========================

class CommonVoicePhonemeDataset(Dataset):
    def __init__(self, dataframe, audio_dir, processor):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = os.path.join(self.audio_dir, row["path"])

        try:
            waveform, sr = librosa.load(audio_path, sr=16000)
            waveform = torch.tensor(waveform)
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return a dummy sample in case of error
            waveform = torch.zeros(16000)

        input_values = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.squeeze()

        with self.processor.as_target_processor():
            labels = self.processor(row["phonemes"]).input_ids

        return {
            "input_values": input_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# =========================
# Data Collator
# =========================

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id,
            -100
        )

        batch["labels"] = labels
        return batch


# =========================
# Metrics (PER)
# =========================

wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)

    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    per = wer_metric.compute(predictions=pred_str, references=label_str)

    # Additional metrics for alignment quality
    avg_unique = np.mean([len(set(seq)) for seq in pred_ids])

    # Check for blank-heavy predictions (sign of poor alignment)
    blank_ratio = np.mean([
        np.sum(seq == processor.tokenizer.pad_token_id) / len(seq)
        for seq in pred_ids
    ])

    return {
        "per": per,
        "avg_unique_phonemes": avg_unique,
        "blank_ratio": blank_ratio
    }


# =========================
# Safety Callback
# =========================

class SafetyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Check for NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"NaN/Inf gradient in {name} at step {state.global_step}")
                    control.should_training_stop = True
                    return control

        # Log gradient norm
        if state.global_step % 50 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 100:
                logger.warning(f"Step {state.global_step}: High gradient norm = {total_norm:.2f}")
            else:
                logger.info(f"Step {state.global_step}: Gradient norm = {total_norm:.2f}")

        return control


# =========================
# Main
# =========================

def main(args):
    logger.info("Loading phoneme CSV...")
    df = pd.read_csv(args.csv_path)

    # Remove any rows with missing or invalid data
    initial_len = len(df)
    df = df.dropna(subset=['path', 'phonemes'])
    df = df[df['phonemes'].str.strip() != '']
    logger.info(f"Removed {initial_len - len(df)} invalid samples")
    logger.info(f"Total samples after cleaning: {len(df)}")

    logger.info("Splitting train/validation...")
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Validation size: {len(val_df)}")

    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = Wav2Vec2CTCTokenizer(
        args.vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    global processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    # Verify phoneme tokenization
    sample_phonemes = train_df['phonemes'].iloc[0]
    logger.info(f"Sample phonemes: {sample_phonemes}")
    tokens = tokenizer(sample_phonemes).input_ids
    logger.info(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokens)
    logger.info(f"Decoded: {decoded}")

    # Model
    logger.info("Loading pretrained model...")

    # Try loading from local path first, fallback to HuggingFace
    model_path = args.model_path if os.path.exists(args.model_path) else "LeBenchmark/wav2vec2-FR-7K-large"
    logger.info(f"Loading model from: {model_path}")

    model = Wav2Vec2ForCTC.from_pretrained(
        model_path,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
    )

    logger.info(f"Model vocab size: {model.config.vocab_size}")

    # CRITICAL: Reinitialize CTC head with proper scaling
    logger.info("Reinitializing CTC head...")
    std = 0.01
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=std)
    if model.lm_head.bias is not None:
        nn.init.zeros_(model.lm_head.bias)

    # Check CTC head initialization
    logger.info(f"CTC head weight stats: mean={model.lm_head.weight.mean():.6f}, "
                f"std={model.lm_head.weight.std():.6f}, "
                f"max={model.lm_head.weight.max():.6f}, "
                f"min={model.lm_head.weight.min():.6f}")

    # Freeze the CNN feature extractor
    logger.info("Freezing feature extractor (CNN layers)...")
    model.freeze_feature_encoder()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Safety check for NaN/Inf in model parameters
    logger.info("Checking model parameters for NaN/Inf...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/Inf found in parameter: {name}")
    logger.info("Model parameters are valid.")

    # Datasets
    logger.info("Creating datasets...")
    train_dataset = CommonVoicePhonemeDataset(
        train_df,
        args.audio_dir,
        processor
    )

    val_dataset = CommonVoicePhonemeDataset(
        val_df,
        args.audio_dir,
        processor
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Test a sample batch
    logger.info("Testing data loading...")

    test_loader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
    sample_batch = next(iter(test_loader))
    logger.info(
        f"Sample batch - Input shape: {sample_batch['input_values'].shape}, Labels shape: {sample_batch['labels'].shape}")
    logger.info(f"Input values stats: min={sample_batch['input_values'].min():.4f}, "
                f"max={sample_batch['input_values'].max():.4f}, "
                f"mean={sample_batch['input_values'].mean():.4f}")

    # Check for data anomalies
    if torch.isnan(sample_batch['input_values']).any():
        raise ValueError("NaN detected in input values!")
    if (sample_batch['labels'] < -100).any():
        raise ValueError("Invalid label values detected!")

    # ==========================================
    # STAGE 1: Train only CTC head
    # ==========================================
    logger.info("=" * 70)
    logger.info("STAGE 1: Training CTC head only (transformer frozen)")
    logger.info("=" * 70)

    # Freeze transformer encoder
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage 1 - Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Stage 1 training arguments
    stage1_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "stage1"),

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        max_steps=2000,  # Fixed number of steps

        learning_rate=1e-4,  # Higher LR for CTC head only
        warmup_steps=200,
        lr_scheduler_type="linear",

        max_grad_norm=1.0,

        fp16=True,
        bf16=False,

        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,

        weight_decay=0.01,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs_stage1"),
        logging_nan_inf_filter=False,
    )

    stage1_trainer = Trainer(
        model=model,
        args=stage1_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[SafetyCallback()],
    )

    logger.info("Starting Stage 1 training...")
    try:
        stage1_trainer.train()
    except Exception as e:
        logger.error(f"Stage 1 training failed: {e}")
        raise

    logger.info("Stage 1 complete. Saving checkpoint...")
    stage1_trainer.save_model(os.path.join(args.output_dir, "stage1_final"))

    # ==========================================
    # STAGE 2: Unfreeze transformer and fine-tune
    # ==========================================
    logger.info("=" * 70)
    logger.info("STAGE 2: Fine-tuning full model (transformer unfrozen)")
    logger.info("=" * 70)

    # Unfreeze transformer encoder
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Stage 2 - Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Stage 2 training arguments - MUCH lower learning rate
    stage2_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "stage2"),

        per_device_train_batch_size=4,  # Smaller batch for stability
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        num_train_epochs=10,

        learning_rate=3e-6,  # VERY low LR for fine-tuning
        warmup_steps=1000,
        lr_scheduler_type="cosine",

        max_grad_norm=0.5,  # More aggressive clipping

        fp16=True,
        bf16=False,

        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,

        weight_decay=0.01,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs_stage2"),
        logging_nan_inf_filter=False,
    )

    stage2_trainer = Trainer(
        model=model,
        args=stage2_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[SafetyCallback()],
    )

    logger.info("Starting Stage 2 training...")
    try:
        stage2_trainer.train()
    except Exception as e:
        logger.error(f"Stage 2 training failed: {e}")
        raise

    logger.info("=" * 70)
    logger.info("Two-stage training complete!")
    logger.info("=" * 70)

    # Save final model
    logger.info("Saving final model...")
    stage2_trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))

    logger.info(f"Model saved to {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/commonvoice_fr_50k_phonemes.csv",
        help="Path to the CSV file containing phoneme annotations"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/vol/corpora/CommonVoice/cv-corpus-19.0-2024-09-13/fr/clips",
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="utils/vocab.json",
        help="Path to the vocabulary JSON file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wav2vec2-FR-7K-large",
        help="Path to pretrained model (local or HuggingFace)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/wav2vec2-fr-phoneme",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    args = parser.parse_args()

    main(args)