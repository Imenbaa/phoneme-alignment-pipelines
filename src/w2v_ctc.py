import torch
import numpy as np
import pandas as pd
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
import json
from tqdm import tqdm
import os
from textgrid import TextGrid
import editdistance
from utils.VAD_chunk import *
import Levenshtein
import librosa
from jiwer import process_words

import unicodedata

REF_MAPPING = {
    # Consonants
    "Z": "ʒ",
    "S": "ʃ",
    "R": "ʁ",
    "N": "ŋ",
    "J": "ɲ",
    "H": "ɥ",
    "g": "ɡ",
    "Z=": "ʒ",

    # Vowels
    "E": "ɛ",
    "O": "ɔ",
    "2": "ø",
    "9": "œ",
    "@": "ə",

    # Nasals (SAMPA)
    "a~": "ɑ̃",
    "o~": "ɔ̃",
    "e~": "ɛ̃",
    "9~": "ɛ̃",
    "E": "ɛ",
    "m=": "m",
    "n=": "n",
    "9~": "ɛ̃",
}
NASAL_CANONICAL = {

    "ã": "ɑ̃",
    "ẽ": "ɛ̃",
    "ĩ": "ɛ̃",
    "ỹ": "ɛ̃",
    "œ̃": "ɛ̃",
    "ə̃": "ɛ̃",
}
HYP_PROJECTION = {

    # Multilingual vowel variants
    "ɪ": "i",
    "ʊ": "u",
    "ɨ": "i",
    "ɜ": "ə",
    "ʌ": "ɔ",
    "ɒ": "ɔ",

    # Rhotic variants
    "ɣ": "ʁ",
    "ɹ": "ʁ",
    "ɾ": "ʁ",

    # Lateral variant
    "ʎ": "l",

    # Foreign consonants
    "β": "b",
    "θ": "t",
    "c": "k",
    "ɑ": "a",
    "mʲ": "m",
    "ɟ": "ɡ",
}
VOWELS = set("aeiouyɛøœɔɑɨɪʊʌɒɜəɛ")
foreign_phonemes = {
    'β','θ','ɹ','ɾ','ɣ','ʌ','ʊ','ɪ','ɨ','ɨ̃','ɜ','ɒ','õ','ũ'
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


        ref_alignments.append({
            "phoneme": phoneme,
            "start": interval.minTime,
            "end": interval.maxTime
        })
    return ref_alignments

def align_sequences(ref, hyp):
    alignment = []
    ops = Levenshtein.editops(ref, hyp)

    ref_idx = hyp_idx = 0
    op_idx = 0

    while ref_idx < len(ref) or hyp_idx < len(hyp):

        if op_idx < len(ops):
            op_type, src_pos, dest_pos = ops[op_idx]

            if op_type == "delete" and src_pos == ref_idx:
                alignment.append((ref_idx, None))
                ref_idx += 1
                op_idx += 1
                continue

            elif op_type == "insert" and dest_pos == hyp_idx:
                alignment.append((None, hyp_idx))
                hyp_idx += 1
                op_idx += 1
                continue

            elif op_type == "replace" and \
                 src_pos == ref_idx and \
                 dest_pos == hyp_idx:
                alignment.append((ref_idx, hyp_idx))
                ref_idx += 1
                hyp_idx += 1
                op_idx += 1
                continue

        # equal case
        if ref_idx < len(ref) and hyp_idx < len(hyp):
            alignment.append((ref_idx, hyp_idx))
            ref_idx += 1
            hyp_idx += 1
        elif ref_idx < len(ref):
            alignment.append((ref_idx, None))
            ref_idx += 1
        elif hyp_idx < len(hyp):
            alignment.append((None, hyp_idx))
            hyp_idx += 1

    return alignment


def match_alignments_lev(ref_alignments, hyp_alignments, ref_seq, hyp_seq):
    alignment = align_sequences(ref_seq, hyp_seq)
    # cm = confusion_matrix_from_alignment(ref_seq, hyp_seq, alignment)
    start_errors = []
    end_errors = []
    duration_errors = []

    for ref_idx, hyp_idx in alignment:

        if ref_idx is None or hyp_idx is None:
            continue

        if ref_seq[ref_idx] == hyp_seq[hyp_idx]:
            r_start = ref_alignments[ref_idx]["start"]
            r_end = ref_alignments[ref_idx]["end"]

            h_start = hyp_alignments[hyp_idx]["start"]
            h_end = hyp_alignments[hyp_idx]["end"]

            start_errors.append(abs(r_start - h_start))
            end_errors.append(abs(r_end - h_end))
            duration_errors.append(
                abs((r_end - r_start) - (h_end - h_start))
            )

    return start_errors, end_errors, duration_errors




def get_phoneme_alignments(model, processor, audio_path):
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(audio)
    chunks = vad_chunk_with_timestamps(wav)
    device = next(model.parameters()).device
    blank_id = model.config.pad_token_id

    all_alignments = []
    full_phoneme_parts = []

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
        full_phoneme_parts.append(decoded.strip())

        num_frames = logits.shape[1]
        chunk_duration = (end_sample - start_sample) / 16000
        frame_duration = chunk_duration / num_frames

        # ✅ Reset per chunk
        prev_id = None
        current_alignment = None

        for frame_idx, token_id in enumerate(predicted_ids.tolist()):

            if token_id == blank_id:
                prev_id = token_id
                continue

            phoneme = processor.decode([token_id])

            # Skip empty / space / word separator
            if phoneme in ["", " ", "|"]:
                prev_id = token_id
                continue

            # If this is a combining mark → attach to previous phoneme
            if unicodedata.combining(phoneme):
                if current_alignment is not None:
                    current_alignment["phoneme"] += phoneme
                prev_id = token_id
                continue

            if token_id != prev_id:
                if current_alignment is not None:
                    current_alignment["end"] = (
                            start_sample / 16000 + frame_idx * frame_duration
                    )

                start_time = start_sample / 16000 + frame_idx * frame_duration
                current_alignment = {
                    "phoneme": phoneme,
                    "start": start_time,
                    "end": None
                }
                all_alignments.append(current_alignment)

            prev_id = token_id

        # close last phoneme of this chunk using frame-based end
        if current_alignment is not None and current_alignment["end"] is None:
            current_alignment["end"] = (
                    start_sample / 16000 + num_frames * frame_duration
            )
    full_phoneme_string = " ".join(full_phoneme_parts)
    return full_phoneme_string.strip(), all_alignments
def normalize_phoneme(ph, is_hyp=False):
    if ph is None:
        return None

    # 🔥 Strip first
    ph = ph.strip()

    # Unicode normalization
    ph = unicodedata.normalize("NFC", ph)

    # Remove silence / junk
    if ph in {"_", "sil", "spn", "%", "?", "0", "=", "fe~", "Ra~", "sjo~", "~e"}:
        return None

    # Remove standalone combining marks
    if ph and all(unicodedata.combining(c) for c in ph):
        return None

    # Reference mapping
    if not is_hyp and ph in REF_MAPPING:
        ph = REF_MAPPING[ph]

    # Hyp projection
    if is_hyp and ph in HYP_PROJECTION:
        ph = HYP_PROJECTION[ph]

    # Collapse multiple nasal marks
    ph = re.sub(r"\u0303+", "\u0303", ph)

    # Prevent nasalized consonants
    if len(ph) > 1:
        base = ph[0]
        if base not in VOWELS:
            ph = base

    # Canonical nasal mapping
    if ph in NASAL_CANONICAL:
        ph = NASAL_CANONICAL[ph]

    # Remove suprasegmentals
    ph = ph.replace("ː", "")

    # 🔥 Remove foreign AFTER projection
    if ph in foreign_phonemes:
        return None

    if ph == "":
        return None

    return ph


def clean_alignment_dict(alignment_list):
    """
    Normalize phonemes and remove empty or deleted ones.
    Keeps timestamps aligned.
    """

    cleaned = []

    for item in alignment_list:
        phoneme = item["phoneme"]

        # Normalize
        phoneme_norm = normalize_phoneme(phoneme)

        # Remove empty phonemes after normalization
        if phoneme_norm == "" or phoneme_norm is None:
            continue

        cleaned.append({
            "phoneme": phoneme_norm,
            "start": item["start"],
            "end": item["end"]
        })

    return cleaned
def extract_phoneme_sequence(alignment_list):
    return [item["phoneme"] for item in alignment_list]

MODEL_ID = "/vol/experiments3/imbenamor/TAPAS-FRAIS/models/wav2vec2-french-phonemizer"

model = AutoModelForCTC.from_pretrained(MODEL_ID)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

device = "cuda"
model = model.to(device)
model.eval()

audio_dir = "/vol/corpora/Rhapsodie/wav16k_corrected"
textgrid_dir = "/vol/corpora/Rhapsodie/TextGrids-fev2013/"

csv_path = "/vol/experiments3/imbenamor/TAPAS-FRAIS/data/rhap_phonemes_ref.csv"

df = pd.read_csv(csv_path)
all_pairs = []
inventory = set()

total_edits = 0
total_ref_tokens = 0
all_boundary_errors = []
total_S = total_D = total_I = total_N = 0
for _, row in tqdm(df.iterrows(), total=len(df)):

    audio_path = os.path.join(audio_dir, row['audio_filename'])
    if "D0001" not in audio_path and "M2006" not in audio_path and "D1003" not in audio_path:
        textgrid_path = os.path.join(textgrid_dir, row["audio_filename"].replace(".wav", "-Pro.TextGrid"))
        pred_phonemes, pred_alignments = get_phoneme_alignments(model, processor, audio_path)
        ref_alignments = get_reference_alignments(textgrid_path)
        ref_offset = ref_alignments[0]["start"]
        hyp_offset = pred_alignments[0]["start"]

        clean_hyp = clean_alignment_dict(pred_alignments)
        clean_ref = clean_alignment_dict(ref_alignments)
        ref_intervals = [{
            "phoneme": item["phoneme"],
            "start": item["start"] - ref_offset,
            "end": item["end"] - ref_offset} for item in clean_ref]

        hyp_intervals = [{
            "phoneme": item["phoneme"],
            "start": item["start"] - hyp_offset,
            "end": item["end"] - hyp_offset} for item in clean_hyp]
        # Normalize
        ref_seq = extract_phoneme_sequence(clean_ref)
        hyp_seq = extract_phoneme_sequence(clean_hyp)
        all_pairs.append((ref_seq, hyp_seq))
        out = process_words(" ".join(ref_seq), " ".join(hyp_seq))

        total_S += out.substitutions
        total_D += out.deletions
        total_I += out.insertions
        total_N += len(ref_seq)

        start_err, end_err, dur_err = match_alignments_lev(ref_intervals, hyp_intervals, ref_seq, hyp_seq)

        all_boundary_errors.extend(start_err)

        if len(start_err) > 0:
            print("Max file boundary error (sec):", max(start_err))
corpus_per =  (total_S + total_D + total_I) / total_N
if len(all_boundary_errors) > 0:
    #flat_errors = [e for sublist in all_boundary_errors for e in sublist]
    errors = np.array(all_boundary_errors)
    print("GLOBAL max error (sec):", np.max(errors))
    print("GLOBAL median error (sec):", np.median(errors))

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
