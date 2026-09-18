import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_from_disk
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)
import json

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC

# =========================
# Model with VAD
# =========================
class Wav2Vec2ForCTCWithVAD(Wav2Vec2ForCTC):

    def apply_vad_bias(self, logits, vad):
        vad = F.interpolate(vad.unsqueeze(1), size=logits.shape[1], mode="nearest").squeeze(1)
        vad = vad.clamp(1e-4, 1 - 1e-4)

        V = logits.shape[-1]

        sil_mask = torch.zeros(V, device=logits.device)
        sil_mask[self.config.sil_token_id] = 1.0
        sil_mask = sil_mask.view(1, 1, V)

        phoneme_mask = 1 - sil_mask
        vad = vad.unsqueeze(-1)

        logits = logits + (
            sil_mask * torch.log(1 - vad) +
            phoneme_mask * torch.log(vad)
        )
        return logits

    def forward(self, input_values, vad=None):
        outputs = self.wav2vec2(input_values)
        logits = self.lm_head(outputs.last_hidden_state)

        if vad is not None:
            logits = self.apply_vad_bias(logits, vad)

        return {"logits": logits}


# =========================
# Viterbi alignment
# =========================
def build_eY(tokens):

    return tokens


def viterbi_align(log_probs, eY_ids):
    T, V = log_probs.shape
    L = len(eY_ids)

    A = torch.full((T, L), -1e9, device=log_probs.device)
    B = torch.zeros((T, L), dtype=torch.long, device=log_probs.device)

    A[0, 0] = log_probs[0, eY_ids[0]]

    for t in range(1, T):
        for j in range(L):
            candidates = []

            # stay
            candidates.append((A[t-1, j], j))

            # j-1
            if j > 0:
                candidates.append((A[t-1, j-1], j-1))

            # j-2
            if j > 1 and eY_ids[j] != eY_ids[j-2]:
                candidates.append((A[t-1, j-2], j-2))

            best_score, best_state = max(candidates, key=lambda x: x[0])

            A[t, j] = best_score + log_probs[t, eY_ids[j]]
            B[t, j] = best_state

    # backtrack
    path = []
    j = L - 1
    for t in reversed(range(T)):
        path.append(eY_ids[j])
        j = B[t, j]

    return list(reversed(path))

# ========================= 
# Extract timestamps 
# ========================= 
def extract_segments(path_ids, tokenizer, frame_duration=0.02): 
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in path_ids] 
    segments = [] 
    current = tokens[0] 
    start = 0 
    for i, t in enumerate(tokens): 
        if t != current: 
            segments.append({ "token": current, "start": start * frame_duration, "end": i * frame_duration })
            current = t 
            start = i 
    segments.append({ "token": current, "start": start * frame_duration, "end": len(tokens) * frame_duration }) 
    return segments

#adaptive VAD threshold
def compute_adaptive_threshold(vad, percentile=20):
    thr = np.percentile(vad, percentile)
    return max(0.02, min(thr, 0.3))  # clamp for stability
def filter_sil_with_vad(segments, vad, frame_duration=0.02, threshold=0.1):
    filtered = []

    for seg in segments:
        if seg["token"] != "SIL":
            filtered.append(seg)
            continue

        start_f = int(seg["start"] / frame_duration)
        end_f = int(seg["end"] / frame_duration)

        vad_segment = vad[start_f:end_f]
        if len(vad_segment) == 0:
            continue

        # 🔥 robust rule
        silence_ratio = (vad_segment < threshold).mean()

        if silence_ratio > 0.2 and (seg["end"] - seg["start"]) > 0.04:
            filtered.append(seg)

    return filtered

def merge_phonemes_and_silence(phonemes, silences):
    merged = []

    # First add phonemes
    for p in phonemes:
        merged.append(p)

    # Then add silence ONLY where no phoneme exists
    for s in silences:
        overlap = False

        for p in phonemes:
            if not (s["end"] <= p["start"] or s["start"] >= p["end"]):
                overlap = True
                break

        if not overlap:
            merged.append(s)

    # Sort everything
    merged = sorted(merged, key=lambda x: x["start"])

    return merged
def vad_to_silence_segments(vad, frame_duration=0.02, threshold=0.5):
    sil_segments = []
    in_sil = False
    start = 0
    vad = vad.detach().cpu().squeeze() 
    for i, v in enumerate(vad):
        v = float(v)
        if v < threshold and not in_sil:
            start = i
            in_sil = True
        elif v >= threshold and in_sil:
            sil_segments.append({
                "token": "SIL",
                "start": start * frame_duration,
                "end": i * frame_duration
            })
            in_sil = False

    if in_sil:
        sil_segments.append({
            "token": "SIL",
            "start": start * frame_duration,
            "end": len(vad) * frame_duration
        })

    return sil_segments
def compute_vad_energy(audio, frame_size=320, hop_size=320):
    audio = np.array(audio)

    energy = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i+frame_size]
        e = np.mean(frame ** 2)
        energy.append(e)

    energy = np.array(energy)
    energy = energy / (energy.max() + 1e-8)

    return energy  # continuous VAD
# =========================
# MAIN
# =========================

exp_name = "w2vCTC_VAD"
checkpoint = f"results/{exp_name}/checkpoint-15625"  # ← put best checkpoint here
dataset_path = f"results/{exp_name}/preprocessed"
vocab_name = f"vocab_{exp_name}.json"

# Load processor
tokenizer = Wav2Vec2CTCTokenizer(vocab_file=vocab_name, pad_token="[PAD]", unk_token="[UNK]")
feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor, tokenizer)

# Load model
model = Wav2Vec2ForCTCWithVAD.from_pretrained(checkpoint)
model.config.sil_token_id = tokenizer.convert_tokens_to_ids("SIL")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
dataset = load_from_disk(dataset_path)["test"]


results_dict = {}

for idx in range(len(dataset)):
    sample = dataset[idx]

    # 🔹 recover filename (adapt depending on your dataset)
    if "path" in sample:
        filename = sample["path"]
    elif "file" in sample:
        filename = sample["file"]
    else:
        filename = f"sample_{idx}"

    input_values = torch.tensor(sample["input_values"]).unsqueeze(0).to(device)
    # ===== Compute VAD =====
    audio = sample["input_values"]
    
    
    # ===== Adaptive threshold =====
    #threshold = compute_adaptive_threshold(vad_np, percentile=20)
    #print("Threshold:", threshold)
    vad_np = compute_vad_energy(audio)
    vad = torch.tensor(vad_np).unsqueeze(0).to(device)
    
    # ===== Forward =====
    with torch.no_grad():
        logits = model(input_values, vad=vad)["logits"][0]
    
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # ===== Decode tokens =====
    pred_ids = torch.argmax(logits, dim=-1)
    
    pred_tokens = [
        tokenizer.convert_ids_to_tokens(int(i))
        for i in pred_ids
    ]
    
    # CTC collapse
    def ctc_collapse(tokens):
        out = []
        prev = None
        for t in tokens:
            if t != prev and t != "[PAD]":
                out.append(t)
            prev = t
        return out
    
    pred_tokens = ctc_collapse(pred_tokens)
    pred_tokens = [t for t in pred_tokens if t not in ["[PAD]", "[UNK]"]]
    
    # ===== Viterbi =====
    eY = build_eY(pred_tokens)
    eY_ids = [tokenizer.convert_tokens_to_ids(t) for t in eY]
    
    aligned_ids = viterbi_align(log_probs, eY_ids)
    segments = extract_segments(aligned_ids, tokenizer)
    sil_segments = vad_to_silence_segments(vad)
    # ===== Filter SIL using VAD =====
    #segments = filter_sil_with_vad(segments, vad_np,frame_duration=0.02,threshold=threshold)
    
    # ===== Merge =====
    
    segments = merge_phonemes_and_silence(segments, sil_segments)
    results_dict[filename] = segments

    if idx % 100 == 0:
        print(f"{idx}/{len(dataset)} processed")
    

output_path = f"results/{exp_name}/alignment.json"

with open(output_path, "w") as f:
    json.dump(results_dict, f, indent=4)

print(f"Alignment saved to {output_path}")