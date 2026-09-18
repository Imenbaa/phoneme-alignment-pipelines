import torch
import numpy as np
import pandas as pd
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
import argparse
import json
from tqdm import tqdm
import os
from textgrid import TextGrid, IntervalTier,PointTier

import editdistance
from utils.VAD_chunk import *

import unicodedata

############################################
# MAPPINGS
############################################
#SAMBA to IPA
ref_mapping = {
    "a": "a",
    "b": "b",
    "c": "k",
    "d": "d",
    "Z": "ʒ",
    "e": "e",
    "f": "f",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "n": "n",
    "o": "o",
    "p": "p",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
    "2": "ø",
    "9": "œ",
    "N": "ŋ",
    "@": "ə",
    "E": "ɛ",
    "O": "ɔ",
    "R": "ʁ",
    "S": "ʃ",
    "J": "ɲ",
    "H": "ɥ",
    "g": "ɡ",
    "g":"ɡ",

    # Nasals
    "a~": "ɑ̃",
    "o~": "ɔ̃",
    "e~": "ɛ̃",
    "9~": "ɛ̃",
}
#IPA to SAMBA
hyp_mapping = {

    # Multilingual vowel projection
    "ɪ": "i",
    "ʊ": "u",
    "ʌ": "ɔ",
    "ɜ": "ə",
    "ɨ": "i",

    # French mergers
    "ɑ": "a",
    "ɒ": "ɔ",

    # Rhotic variants
    "ɣ": "ʁ",
    "ɹ": "ʁ",
    "ɾ": "ʁ",

    # Palatal lateral
    "ʎ": "l",

    # Palatalized consonants
    "mʲ": "m",

    # Affricates collapse
    "tʃ": "ʃ",
    "ts": "s",

    # Greek / foreign consonants
    "β": "b",
    "θ": "s",

    # NASAL VOWELS → match reference inventory
    "ã": "ɑ̃",
    "ẽ": "ɛ̃",
    "ĩ": "ɛ̃",
    "õ": "ɔ̃",
    "ũ": "ɔ̃",
    "ỹ": "ɛ̃",
}


def get_reference_alignments(textgrid_path):

    tg = TextGrid()
    tg.read(textgrid_path)

    ref_alignments = []

    # assuming tier name is "phones"
    tier = tg.getFirst("phone")

    for interval in tier.intervals:

        phoneme = interval.mark.strip()

        if phoneme == "" or phoneme in ["sil", "sp", "spn"]:
            continue

        midpoint = (interval.minTime + interval.maxTime) / 2

        ref_alignments.append({
            "phoneme": phoneme,
            "time_sec": midpoint
        })

    return ref_alignments
import difflib

def match_alignments(ref_alignments, pred_alignments):

    # Normalisation phonème par phonème
    ref_seq = [
        normalize_reference(p["phoneme"])
        for p in ref_alignments
    ]

    pred_seq = [
        normalize_hypothesis(p["phoneme"])
        for p in pred_alignments
    ]

    matcher = difflib.SequenceMatcher(None, ref_seq, pred_seq)

    boundary_errors = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            for r, p in zip(range(i1, i2), range(j1, j2)):
                error = abs(
                    ref_alignments[r]["time_sec"] -
                    pred_alignments[p]["time_sec"]
                )
                boundary_errors.append(error)

    return boundary_errors

############################################
# NORMALIZATION FUNCTIONS
############################################

def normalize_reference(ref_string):

    ref_string = unicodedata.normalize("NFC", ref_string)

    # Replace longest first (nasals first)
    for k in sorted(ref_mapping.keys(), key=len, reverse=True):
        ref_string = ref_string.replace(k, ref_mapping[k])

    # Remove noise
    for noise in ["_", "sil", "spn", "%", "?", "0", "="]:
        ref_string = ref_string.replace(noise, "")

    # Remove combining tilde if any left
    ref_string = ref_string.replace("̃", "")

    ref_string = ref_string.replace(" ", "")

    return ref_string


def normalize_hypothesis(decoded_string):


    hyp = unicodedata.normalize("NFC", decoded_string)
    hyp = hyp.replace(" ", "")

    for k in sorted(hyp_mapping.keys(), key=len, reverse=True):
        hyp = hyp.replace(k, hyp_mapping[k])

    # remove length markers
    hyp = hyp.replace("ː", "")

    # remove stray combining tilde
    hyp = hyp.replace("̃", "")

    return hyp


############################################
# PER FUNCTION (CHARACTER-LEVEL)
############################################

def compute_per(ref_string, hyp_string):

    ref_tokens = list(ref_string)
    hyp_tokens = list(hyp_string)

    if len(ref_tokens) == 0:
        return 0, 0

    distance = editdistance.eval(ref_tokens, hyp_tokens)

    return distance, len(ref_tokens)


############################################
# PHONEME + ALIGNMENT
############################################

def get_phoneme_alignments(model, processor, audio_path):

    audio, sr = sf.read(audio_path)

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    wav = torch.from_numpy(audio)

    chunks = vad_chunk_with_timestamps(wav)

    device = next(model.parameters()).device
    frame_duration = model.config.inputs_to_logits_ratio / 16000

    all_alignments = []
    full_phoneme_string = ""

    for chunk in chunks:

        start_sample = int(chunk["start"] * 16000)
        end_sample = int(chunk["end"] * 16000)

        chunk_tensor = wav[start_sample:end_sample]

        inputs = processor(
            chunk_tensor.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)[0]

        decoded = processor.batch_decode(predicted_ids.unsqueeze(0))[0]

        full_phoneme_string += decoded

        prev_id = None

        for frame_idx, token_id in enumerate(predicted_ids.tolist()):

            if token_id != processor.tokenizer.pad_token_id and token_id != prev_id:
                phoneme = processor.decode([token_id]).strip()

                local_time = frame_idx * frame_duration
                absolute_time = (start_sample / 16000) + local_time

                all_alignments.append({
                    "phoneme": phoneme,
                    "time_sec": absolute_time
                })

            prev_id = token_id

    return full_phoneme_string.strip(), all_alignments


############################################
# DATASET PROCESSING
############################################

def process_dataset(model, processor, csv_path, audio_dir, json_output_path, csv_output_path,textgrid_dir):

    df = pd.read_csv(csv_path)

    json_results = []       # for phoneme_predictions.json
    csv_rows = []           # for per_results.csv

    total_edits = 0
    total_ref_tokens = 0
    all_boundary_errors=[]

    for _, row in tqdm(df.iterrows(), total=len(df)):

        audio_path = os.path.join(audio_dir, row['audio_filename'])

        try:
            pred_phonemes, pred_alignments = \
                get_phoneme_alignments(model, processor, audio_path)

            # Normalize
            ref_raw = row.get("reference_phonemes", "")
            ref_norm = normalize_reference(ref_raw)
            hyp_norm = normalize_hypothesis(pred_phonemes)

            distance, ref_len = compute_per(ref_norm, hyp_norm)

            if ref_len > 0:
                file_per = distance / ref_len
                total_edits += distance
                total_ref_tokens += ref_len
            else:
                file_per = 0

            # ---- JSON output (keep full alignment info)
            json_results.append({
                "audio_filename": row["audio_filename"],
                "predicted_phonemes": pred_phonemes,
                "alignments": pred_alignments
            })

            # ---- CSV output (PER + phonemes)
            csv_rows.append({
                "audio_filename": row["audio_filename"],
                "file_per": file_per,
                "reference_phonemes_raw": ref_raw,
                "predicted_phonemes_raw": pred_phonemes,
                "reference_phonemes_norm": ref_norm,
                "predicted_phonemes_norm": hyp_norm
            })
            textgrid_path = os.path.join(
                textgrid_dir,
                row["audio_filename"].replace(".wav", "-Pro.TextGrid")
            )

            ref_alignments = get_reference_alignments(textgrid_path)
            print("Ref alignments:", len(ref_alignments))
            print("Pred alignments:", len(pred_alignments))

            boundary_errors = match_alignments(ref_alignments, pred_alignments)

            all_boundary_errors.extend(boundary_errors)


        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    corpus_per = total_edits / total_ref_tokens if total_ref_tokens > 0 else 0



    if len(all_boundary_errors) > 0:
        errors = np.array(all_boundary_errors)

        mean_boundary_error = np.mean(errors)
        median_boundary_error = np.median(errors)

        within_20ms = np.mean(errors <= 0.02) * 100
        within_50ms = np.mean(errors <= 0.05) * 100
    else:
        mean_boundary_error = 0
        median_boundary_error = 0
        within_20ms = 0
        within_50ms = 0
    print("\n===== FINAL RESULTS =====")
    print(f"Corpus PER: {corpus_per * 100:.2f}%")
    print(f"Mean boundary error: {mean_boundary_error * 1000:.2f} ms")
    print(f"Median boundary error: {median_boundary_error * 1000:.2f} ms")
    print(f"% within 20ms: {within_20ms:.2f}%")
    print(f"% within 50ms: {within_50ms:.2f}%")

    # ---- Save JSON (phoneme predictions + timestamps)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    print(f"Phoneme predictions saved to {json_output_path}")

    # ---- Save CSV (PER analysis)
    pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False)

    print(f"PER results saved to {csv_output_path}")


############################################
# MAIN
############################################

def main(args):

    print("Loading model...")

    MODEL_ID = "/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wav2vec2-french-phonemizer"

    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    device = "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    process_dataset(
        model,
        processor,
        args.csv_path,
        args.audio_dir,
        args.json_output,
        args.csv_output,args.textgrid_dir

    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--json_output", type=str,
                        default="./phoneme_predictions.json")
    parser.add_argument("--csv_output", type=str,
                        default="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/per_results.csv")
    parser.add_argument("--textgrid_dir", type=str,
                        default="/vol/corpora/Rhapsodie/TextGrids-fev2013")
    args = parser.parse_args()

    main(args)
