import os

os.environ["HF_HOME"] = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"] = "/vol/experiments/cache_imbenamor/datasets"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import numpy as np
import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from typing import Dict, List, Union
import editdistance
from pyannote.audio import Model
from datasets import load_dataset, load_from_disk, Audio as DatasetAudio
from pyannote.audio.core.io import Audio
# =========================
# Config
# =========================
exp_name = "w2vCTC_VAD_pyannote"
model_name = "/vol/experiments2/cbrazier/transcription/models/models--LeBenchmark--wav2vec2-FR-7K-large/snapshots/7fa1111246fca6eb198d1caab50fd2e4469bf659"
output_dir = f"results/{exp_name}"
os.makedirs(output_dir, exist_ok=True)
vocab_name = f"vocab_{exp_name}.json"
preprocessed_path = f"{output_dir}/preprocessed"

TOKEN = os.environ["HF_TOKEN"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vad_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=os.environ["HF_TOKEN"])
vad_model.to(device)
vad_model.eval()
audio_loader = Audio(sample_rate=16000)


# =========================
# VAD
# =========================
def compute_pyannote_vad_batch(audio_arrays):
    # pad to same length
    max_len = max(len(a) for a in audio_arrays)

    batch = []
    for a in audio_arrays:
        t = torch.tensor(a, dtype=torch.float32)
        if len(t) < max_len:
            t = torch.nn.functional.pad(t, (0, max_len - len(t)))
        batch.append(t)

    batch = torch.stack(batch).to(device)  # (B, T)

    with torch.no_grad():
        scores = vad_model(batch.unsqueeze(1))  # (B, frames, classes)

    vad_list = []
    for i, a in enumerate(audio_arrays):
        vad = scores[i, :, 1].cpu().numpy()

        # crop back to real length
        target_len = int(np.ceil(len(a) / 320))  # 20ms
        vad = vad[:target_len]

        vad = np.clip(vad, 1e-4, 1 - 1e-4)
        vad_list.append(vad.astype(np.float32))

    return vad_list


def resample_vad_to_20ms(vad, audio_len, sr=16000, frame_ms=20):

    target_len = int(np.ceil(audio_len / (sr * frame_ms / 1000)))

    x_old = np.linspace(0, 1, len(vad))
    x_new = np.linspace(0, 1, target_len)

    vad_resampled = np.interp(x_new, x_old, vad)

    return vad_resampled.astype(np.float32)


def compute_energy_vad(audio, sr=16000, frame_ms=20, threshold=0.005):
    frame_size = int(sr * frame_ms / 1000)
    vad = []
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        if len(frame) == 0:
            continue
        energy = np.mean(np.abs(frame))  # average amplitude
        vad.append(0.0 if energy < threshold else 1.0)
    return np.array(vad,
                    dtype=np.float32)  # output is a vector or 0 and 1 of length = number of frames (1value is 20ms)


# =========================
# Insert SIL
# =========================
def insert_silence(seq):
    tokens = seq.split()
    result = ["SIL"]
    for t in tokens:
        result.append(t)
        result.append("SIL")
    return " ".join(result)


# =========================
# Dataset
# =========================
if os.path.exists(preprocessed_path):
    dataset = load_from_disk(preprocessed_path)
else:
    dataset = load_dataset(
        "audiofolder",
        data_dir="/vol/experiments2/cbrazier/transcription/datasets/ester1_ester2_epac"
    )
    dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", DatasetAudio(sampling_rate=16000))


    def has_no_unk(example):
        return "|ø|n|k|" not in example["phonemes"]


    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].filter(has_no_unk, num_proc=6)
    # vocab
    vocab_set = set()
    for split in ["train", "validation", "test"]:
        for seq in dataset[split]["phoneme_single_v1"]:
            vocab_set.update(seq.split())

    vocab_list = sorted(vocab_set)
    vocab_dict = {"[PAD]": 0, "[UNK]": 1, "SIL": 2}
    vocab_dict.update({v: i + 3 for i, v in enumerate(vocab_list)})

    with open(vocab_name, "w") as f:
        json.dump(vocab_dict, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_name,
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


    def preprocess_batch(batch):
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == 16000 for sr in sampling_rates), "Unexpected sampling rate"

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="np",
            padding=False
        )

        phonemes = [s for s in batch["phoneme_single_v1"]]

        labels = processor.tokenizer(phonemes).input_ids

        vad = compute_pyannote_vad_batch(audio_arrays)

        return {
            "input_values": inputs.input_values,
            "labels": labels,
            "vad": vad,
            "length": [len(a) for a in audio_arrays],
        }


    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].map(
            preprocess_batch,
            batched=True,
            batch_size=64,
            num_proc=1,
            remove_columns=dataset[split].column_names,
            cache_file_name=f"/vol/experiments/cache_imbenamor/map_vad_pyannote_{split}.arrow"
        )

    dataset.save_to_disk(preprocessed_path)

# =========================
# Processor reload
# =========================
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_name,
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

sil_id = tokenizer.convert_tokens_to_ids("SIL")


# =========================
# Data collator
# =========================
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
        # VAD padding
        vad = [torch.tensor(f["vad"]) for f in features]
        vad = torch.nn.utils.rnn.pad_sequence(vad, batch_first=True)

        batch["labels"] = labels
        batch["vad"] = vad

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)


# =========================
# Model with VAD
# =========================
class Wav2Vec2ForCTCWithVAD(Wav2Vec2ForCTC):

    def apply_vad_bias(self, logits, vad, alpha=1.0):
        """
        logits (Batch_size, T: nb of timesteps 50/sec, Vocab_size)
        VAD (batch_size, T_vad)
        """
        # align T_vad with T
        vad = torch.nn.functional.interpolate(
            vad.unsqueeze(1),
            size=logits.shape[1],
            mode="nearest"
        ).squeeze(1)  # B,T

        vad = vad.clamp(1e-4, 1 - 1e-4)  # avoid -infinie

        B, T, V = logits.shape

        sil_mask = torch.zeros(V, device=logits.device)
        sil_mask[self.config.sil_token_id] = 1.0
        sil_mask = sil_mask.view(1, 1, V)  # 1,1,V

        vad = vad.unsqueeze(-1)

        sil_bias = torch.log(1 - vad)
        non_sil_bias = torch.log(vad)

        # Apply SIL bias
        logits[:, :, self.config.sil_token_id] += alpha * sil_bias.squeeze(-1)

        # Apply non-SIL bias
        mask = torch.ones_like(logits)
        mask[:, :, self.config.sil_token_id] = 0

        logits += alpha * non_sil_bias * mask
        return logits

    def forward(self, input_values, labels=None, vad=None, **kwargs):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if vad is not None:
            logits = self.apply_vad_bias(logits, vad)

        loss = None

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            input_lengths = torch.full(
                (logits.shape[0],),
                logits.shape[1],
                dtype=torch.long,
                device=logits.device,
            )

            label_mask = labels != -100
            label_lengths = label_mask.sum(-1)

            loss = F.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
                blank=self.config.ctc_blank_token_id,
                zero_infinity=True,
            )

        return {"loss": loss, "logits": logits}


# =========================
# Model init
# =========================
model = Wav2Vec2ForCTCWithVAD.from_pretrained(
    model_name,
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
)

model.config.ctc_loss_reduction = "mean"
model.config.ctc_zero_infinity = True
model.config.ctc_blank_token_id = tokenizer.pad_token_id
model.config.sil_token_id = sil_id
model.freeze_feature_encoder()


# =========================
# Metrics
# =========================
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
        #pred_tokens = [t for t in pred_tokens if t != "SIL"]
        label_tokens = [processor.tokenizer.convert_ids_to_tokens(int(id_)) for id_ in label_id_seq if
                        id_ not in [-100, processor.tokenizer.pad_token_id]]
        all_per.append(editdistance.eval(pred_tokens, label_tokens) / max(1, len(label_tokens)))

    return {"PER": np.mean(all_per)}


# =========================
# Training
# =========================
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
    dataloader_pin_memory=True,
    load_best_model_at_end=True,
    metric_for_best_model="PER",
    greater_is_better=False,
)


class CTCTrainer(Trainer):
    def _get_eval_sampler(self, eval_dataset):
        return None


trainer = CTCTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(output_dir)