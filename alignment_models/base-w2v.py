import os
os.environ["HF_HOME"] = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"] = "/vol/experiments/cache_imbenamor/datasets"
os.environ["TMPDIR"] = "/vol/experiments/cache_imbenamor/tmp"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataclasses import dataclass
from datasets import load_dataset, load_from_disk, Audio
import json
import numpy as np
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Union
import editdistance
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler


# =========================
# Step 0: Config
# =========================
print("Step 0: Choosing metadata")

exp_name = "w2vCTC"
model_name = "/vol/experiments2/cbrazier/transcription/models/models--LeBenchmark--wav2vec2-FR-7K-large/snapshots/7fa1111246fca6eb198d1caab50fd2e4469bf659"
output_dir = f"results/{exp_name}"
os.makedirs(output_dir, exist_ok=True)
vocab_name = f"vocab_{exp_name}.json"
preprocessed_path = f"results/{exp_name}/preprocessed"

# =========================
# Steps 1-6: Preprocessing (cached)
# =========================

if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset from disk...")
    dataset = load_from_disk(preprocessed_path)
    
else:
    print("Preprocessing from scratch...")

    # Step 1: Load dataset
    print("Step 1: Loading dataset")
    dataset = load_dataset(
        "audiofolder",
        data_dir="/vol/experiments2/cbrazier/transcription/datasets/ester1_ester2_epac"
    )
    dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Step 2: Filter UNK
    print("Step 2: Filtering UNK")
    def has_no_unk(example):
        return "|ø|n|k|" not in example["phonemes"]

    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].filter(has_no_unk, num_proc=6)

    # Step 4: Build vocab
    print("Step 4: Build vocab")
    vocab_set = set()
    for split in ["train", "validation", "test"]:
        for seq in dataset[split]["phoneme_single_v1"]:
            vocab_set.update(seq.split())

    vocab_list = sorted(vocab_set)
    vocab_dict = {"[PAD]": 0, "[UNK]": 1}
    vocab_dict.update({v: i + 2 for i, v in enumerate(vocab_list)})

    with open(vocab_name, "w") as f:
        json.dump(vocab_dict, f)

    print(f"Vocab size: {len(vocab_dict)} — {list(vocab_dict.keys())}")

    # Step 5: Processor (needed inside preprocess_batch)
    print("Step 5: Create processor")
    _tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_name,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=""
    )
    _feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    _processor = Wav2Vec2Processor(
        feature_extractor=_feature_extractor,
        tokenizer=_tokenizer
    )

    # Step 6: Preprocessing
    print("Step 6: Preprocessing")
    def preprocess_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == 16000 for sr in sampling_rates), "Unexpected sampling rate"
        
        inputs = _feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="np",
            padding=False
        )
        with _processor.as_target_processor():
            labels = _processor(batch["phoneme_single_v1"]).input_ids

        return {
            "input_values": inputs.input_values,
            "labels": labels,
            "length": [len(a) for a in audio_arrays],
        }

    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].map(
            preprocess_batch,
            batched=True,
            batch_size=32,
            num_proc=6,
            remove_columns=dataset[split].column_names,
            cache_file_name=f"/vol/experiments/cache_imbenamor/map_{split}.arrow"
        )
    # Save to disk for future runs
    print("Saving preprocessed dataset to disk...")
    dataset.save_to_disk(preprocessed_path)

# =========================
# Always load processor
# =========================
print("Loading processor...")
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_name,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token=""
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# =========================
# Step 7: Data collator
# =========================
print("Step 7: Data collator")

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# =========================
# Step 8: Metrics (PER)
# =========================
print("Step 8: Metrics")

def ctc_collapse(token_list):
    collapsed = []
    prev = None
    for t in token_list:
        if t != prev and t != "[PAD]":
            collapsed.append(t)
        prev = t
    return collapsed

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids

    all_per = []
    for pred_id_seq, label_id_seq in zip(pred_ids, label_ids):
        pred_tokens = ctc_collapse([processor.tokenizer.convert_ids_to_tokens(int(id_)) for id_ in pred_id_seq])
        label_tokens = [processor.tokenizer.convert_ids_to_tokens(int(id_)) for id_ in label_id_seq if id_ not in [-100, processor.tokenizer.pad_token_id]]
        all_per.append(editdistance.eval(pred_tokens, label_tokens) / max(1, len(label_tokens)))

    return {"PER": np.mean(all_per)}
# =========================
# Step 9: Model
# =========================
print("Step 9: Model")

model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_encoder()

# =========================
# Step 10: Training
# =========================
print("Step 10: Training")

training_args = TrainingArguments(
    output_dir=output_dir,
    group_by_length=True,
    length_column_name="length",
     per_device_train_batch_size=32,        
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=30,
    learning_rate=1e-5,
    warmup_steps=1000,
    weight_decay=0.005,
    logging_steps=500,
    gradient_checkpointing=True,
    max_grad_norm=2.0,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
)

class CTCTrainer(Trainer):
    def _get_eval_sampler(self, eval_dataset):
        return None

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
trainer.save_model(output_dir)

# =========================
# Save logs
# =========================
logs_path = os.path.join(output_dir, "logs.json")
with open(logs_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)