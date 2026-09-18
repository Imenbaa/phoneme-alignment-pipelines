import os
import json
import argparse
import unicodedata

import logging
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Dict, Union
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
from transformers import (
Wav2Vec2CTCTokenizer,
Wav2Vec2FeatureExtractor,
Wav2Vec2Processor,
WavLMForCTC,
TrainingArguments,
Trainer,
TrainerCallback
)
import evaluate
from torch.utils.data import Dataset
import torch.nn as nn

logging.basicConfig(
format="%(asctime)s | %(levelname)s | %(message)s",
level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
            MAX_DURATION = 12  # seconds
            max_len = int(sr * MAX_DURATION)

            if len(waveform) > max_len:
                waveform = waveform[:max_len]

            waveform = torch.tensor(waveform)

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy sample in case of error
            waveform = torch.zeros(16000)

        input_values = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.squeeze()

        tokens = row["phonemes"].split()
        phoneme_inventory = set(
            token
            for seq in row["phonemes"]
            for token in seq.split()
        )


        labels = []
        for t in tokens:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(t)
            if token_id is None:
                print("❌ UNKNOWN TOKEN:", repr(t))
                print("Full sequence:", row["phonemes"])
                raise ValueError(f"Token {t} not in vocab")
            labels.append(token_id)

        return {
            "input_values": input_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


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



# At the top, replace your basicConfig with:
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),  # terminal
            logging.FileHandler(log_path),  # file
        ]
    )
    return logging.getLogger(__name__)

def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)

    # Decode predictions
    pred_str = processor.batch_decode(pred_ids)

    # Decode labels
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    # 🔥 REMOVE SPACES (critical)
    pred_str = [s.replace(" ", "") for s in pred_str]
    label_str = [s.replace(" ", "") for s in label_str]

    per = cer_metric.compute(predictions=pred_str, references=label_str)

    # Extra diagnostics
    avg_unique = np.mean([len(set(seq)) for seq in pred_ids])

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
class MetricsLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = state.epoch or 0
            # Filter and format relevant metrics
            relevant = {k: v for k, v in logs.items()
                       if any(k.startswith(p) for p in
                              ["loss", "eval_", "per", "learning_rate"])}
            if relevant:
                msg = f"Step {step} | Epoch {epoch:.2f} | " + \
                      " | ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                 else f"{k}={v}" for k, v in relevant.items())
                logger.info(msg)
class SafetyCallback(TrainerCallback):
    def on_backward_end(self, args, state, control, model=None, **kwargs):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        print(f"Step {state.global_step}: Gradient norm = {total_norm:.6f}")

def main(args):
    global logger
    logger = setup_logging(args.output_dir)
    df = pd.read_csv(args.csv_path)
    df["phonemes"] = df["phonemes"].apply(lambda x: unicodedata.normalize("NFC", x))

    df["phonemes"] = df["phonemes"].str.replace("ã", "ɑ̃")
    df["phonemes"] = df["phonemes"].str.replace("õ", "ɔ̃")
    df["phonemes"] = df["phonemes"].str.replace("ẽ", "ɛ̃")
    df["phonemes"] = df["phonemes"].str.replace(r"\s+", " ", regex=True).str.strip()

    # 3️⃣ VERIFY CLEANING
    inventory = set(token for seq in df["phonemes"] for token in seq.split())
    print("Inventory inside script:", inventory)

    # 4️⃣ THEN split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)



    # Tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        args.vocab_path,
        pad_token="[PAD]",
        word_delimiter_token="|",
        do_lower_case=False,
        replace_word_delimiter_char="|",
    )


    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Validation size: {len(val_df)}")


    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        logits = pred.predictions
        pred_ids = np.argmax(logits, axis=-1)

        pred_str = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        # remove spaces
        pred_str = [s.replace(" ", "") for s in pred_str]
        label_str = [s.replace(" ", "") for s in label_str]

        per = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"per": per}

    # Verify phoneme tokenization
    sample_phonemes = train_df['phonemes'].iloc[0]
    logger.info(f"Sample phonemes: {sample_phonemes}")
    tokens = tokenizer(sample_phonemes).input_ids
    logger.info(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokens)
    logger.info(f"Decoded: {decoded}")

    model = WavLMForCTC.from_pretrained("/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wavlm_finetuned_big/checkpoint-11000"
        #"/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wavlm_finetuned_unfrozen/checkpoint-10000",
        #vocab_size=37,
        #pad_token_id=processor.tokenizer.pad_token_id,
        #ctc_loss_reduction="mean",
    )
    # Add these lines here:
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model = model.to("cuda")

    # Unfreeze encoder
    for param in model.wavlm.parameters():
        param.requires_grad = True

    # In a custom forward or via a hook — simpler: just use gradient_checkpointing:
    model.gradient_checkpointing_enable()

    logger.info(f"Model vocab size: {model.config.vocab_size}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")

    # Datasets
    logger.info("Creating datasets...")
    train_dataset = CommonVoicePhonemeDataset(train_df, args.audio_dir, processor)

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

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        # Batch size
        per_device_eval_batch_size=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        eval_accumulation_steps=4,  # ← critical for OOM during eval
        gradient_checkpointing=True,  # ← critical for OOM during train
        fp16=True,
        logging_steps=50,

        learning_rate=1e-6,  # 10x smaller than current
        num_train_epochs=3,  # short run
        warmup_steps=200,
        max_grad_norm=1,
        lr_scheduler_type="cosine",


        # Mixed precision
        bf16=False,

        # Evaluation & saving
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,

        # Regularization
        weight_decay=0.01,)

        # Data loading



    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        callbacks=[SafetyCallback(), MetricsLoggerCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))

    logger.info(f"Model saved to {os.path.join(args.output_dir, 'final_model')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        #default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/commonvoice_fr_wavlm.csv",
        default ="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/commonvoice_fr_big_wavlm.csv",
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
        default = "/vol/experiments3/imbenamor/TAPAS-FRAIS/src/utils/vocab.json",
        help="Path to the vocabulary JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wavlm_finetuned_big",
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