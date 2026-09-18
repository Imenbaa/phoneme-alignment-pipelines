import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
import numpy as np
import torch
import argparse
from collections import Counter

from datasets import load_from_disk
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from dataclasses import dataclass
from typing import Dict, List, Union
import editdistance
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC


class Wav2Vec2ForCTCWithVAD(Wav2Vec2ForCTC):

    def apply_vad_bias(self, logits, vad, alpha=1.0):
        vad = torch.nn.functional.interpolate(
            vad.unsqueeze(1),
            size=logits.shape[1],
            mode="nearest"
        ).squeeze(1)

        vad = vad.clamp(1e-4, 1 - 1e-4)

        B, T, V = logits.shape

        sil_mask = torch.zeros(V, device=logits.device)
        sil_mask[self.config.sil_token_id] = 1.0
        sil_mask = sil_mask.view(1, 1, V)

        phoneme_mask = 1 - sil_mask
        vad = vad.unsqueeze(-1)

        logits = logits + alpha * (
                sil_mask * torch.log(1 - vad) +
                phoneme_mask * torch.log(vad)
        )

        return logits

    def forward(self, input_values, vad=None, **kwargs):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        if vad is not None:
            logits = self.apply_vad_bias(logits, vad)

        return {"logits": logits}


# =========================
# Config
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
args = parser.parse_args()

vocab_name = f"vocab_{args.exp_name}.json"
preprocessed_path = f"results/{args.exp_name}/preprocessed"

# =========================
# Load processor
# =========================
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
# Load dataset
# =========================
print("Loading dataset...")
dataset = load_from_disk(preprocessed_path)
test_data = dataset["test"]

# =========================
# Load model
# =========================
#checkpoints = sorted([f"results/{args.exp_name}/" + i for i in os.listdir(f"results/{args.exp_name}/") if i.startswith("checkpoint")],key=lambda x: int(re.findall(r"\d+", x)[-1]))
checkpoints = ["/vol/experiments3/imbenamor/TAPAS-FRAIS/alignment_models/results/w2vCTC_VAD/checkpoint-15625"]

def ctc_collapse(token_list):
    collapsed = []
    prev = None
    for t in token_list:
        if t != prev and t not in ["[PAD]", "[UNK]"]:
            collapsed.append(t)
        prev = t
    return collapsed


for checkpoint_path in checkpoints:
    if "VAD" in args.exp_name:
        model = Wav2Vec2ForCTCWithVAD.from_pretrained(checkpoint_path)
        model.config.sil_token_id = tokenizer.convert_tokens_to_ids("SIL")
    else:
        # run evaluation loop
        print(f"Loading model from {checkpoint_path}...")
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    # =========================
    # Evaluate
    # =========================
    print("Evaluating...")

    all_pred_tokens = []
    all_label_tokens = []
    batch_size = 8

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]

        input_values = [torch.tensor(iv) for iv in batch["input_values"]]
        input_values_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        ).to(device)
        if "VAD" in args.exp_name:
            vad = [torch.tensor(v, dtype=torch.float32) for v in batch["vad"]]
            vad_padded = torch.nn.utils.rnn.pad_sequence(
                vad, batch_first=True, padding_value=0.0
            ).to(device)

            logits = model(input_values_padded, vad=vad_padded)["logits"]
        else:
            logits = model(input_values_padded).logits

        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        # Decode predictions token by token
        for pred_id_seq, label_id_seq in zip(pred_ids, batch["labels"]):
            pred_tokens = [
                processor.tokenizer.convert_ids_to_tokens(int(id_))
                for id_ in pred_id_seq
            ]

            pred_tokens = ctc_collapse(pred_tokens)

            pred_tokens = [t for t in pred_tokens if t not in ["[PAD]", "[UNK]"]]
            label_tokens = [
                processor.tokenizer.convert_ids_to_tokens(int(id_))
                for id_ in label_id_seq
                if id_ not in [processor.tokenizer.pad_token_id, -100]
            ]
            all_pred_tokens.append(pred_tokens)
            all_label_tokens.append(label_tokens)

        if i % 100 == 0:
            print(f"  {i}/{len(test_data)}")

    # =========================
    # Compute PER
    # =========================
    # Print a few examples
    print("\nSample predictions:")
    for i in range(min(5, len(all_pred_tokens))):
        print(f"PRED: {' '.join(all_pred_tokens[i])}")
        print(f"REF:  {' '.join(all_label_tokens[i])}")
        print()
    print("\nDEBUG TOKEN DISTRIBUTION:")


    flat_preds = [t for seq in all_pred_tokens for t in seq]
    print("Pred token counts:", Counter(flat_preds).most_common(10))
    sil_ratio = np.mean([
        sum(1 for t in seq if t == "SIL") / max(1, len(seq))
        for seq in all_pred_tokens
    ])

    print(f"SIL ratio in predictions: {sil_ratio:.3f}")
    per = np.mean([
            editdistance.eval(p, l) / max(1, len(l))
            for p, l in zip(all_pred_tokens, all_label_tokens)
        ])

    print(f"\nPER on test set: {per:.4f}")

    # Save results
    results = {"PER": per, "checkpoint": checkpoint_path}
    with open(f"results/{args.exp_name}/eval_results.json", "a") as f:
        f.write(json.dumps(results) + "\n")
    print("Results saved.")