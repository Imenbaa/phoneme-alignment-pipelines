import torch
import numpy as np
import pandas as pd
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor
import json
import Levenshtein
import numpy as np
import unicodedata
import re
import soundfile as sf
import csv
from praatio import textgrid


from tqdm import tqdm
import os
from metrics import *
from textgrid import TextGrid
import editdistance
import unicodedata
import pickle
from collections import defaultdict
#whisper model
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperModel,
    WhisperPreTrainedModel,
    Wav2Vec2PhonemeCTCTokenizer,
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.modeling_outputs import CausalLMOutput
import math
import torch
import numpy as np
import soundfile as sf
import librosa
import unicodedata
import torchaudio  # >= 2.1 for forced_align

# Whisper encoder: 10 ms mel hop * 2 (conv stride-2) = 20 ms = 320 samples @16 kHz
WHISPER_FRAME_STRIDE_S       = 0.02
WHISPER_FRAME_STRIDE_SAMPLES = 320

def get_phoneme_alignments_whisper_ctcfa(model, feature_extractor, tokenizer,
                                          audio_path, reference=None):
    """
    CTC forced-alignment for a Whisper-encoder + CTC-head phoneme model.
    Now also prints, per chunk, the clean predicted phoneme string. The reference
    (full utterance) is printed once, since VAD chunks have no per-chunk reference.
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(audio).float()

    chunks = vad_chunk_with_timestamps(audio_path)
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype
    blank_id = getattr(model.config, "pad_token_id", None)
    if blank_id is None:
        blank_id = tokenizer.pad_token_id

    if reference is not None:
        print(f"REF (utterance): {reference}")

    all_alignments, full_phoneme_parts = [], []

    for ci, chunk in enumerate(chunks):
        start_sample = int(chunk["start"] * 16000)
        end_sample   = int(chunk["end"]   * 16000)
        chunk_tensor = wav[start_sample:end_sample]
        if chunk_tensor.numel() == 0:
            continue

        inputs = feature_extractor(
            chunk_tensor.numpy(), sampling_rate=16000, return_tensors="pt",
        )
        input_features = inputs.input_features.to(device).to(dtype)

        with torch.no_grad():
            logits = model(input_features=input_features).logits     # (1, ~1500, V)

        valid_frames = max(1, math.ceil(chunk_tensor.shape[0] / WHISPER_FRAME_STRIDE_SAMPLES))
        valid_frames = min(valid_frames, logits.shape[1])
        logits = logits[:, :valid_frames, :]

        predicted_ids = torch.argmax(logits, dim=-1)[0]

        # ---- clean CTC collapse for display (drop repeats AND blanks) ----
        collapsed, prev = [], None
        for t in predicted_ids.tolist():
            if t != prev and t != blank_id:
                collapsed.append(t)
            prev = t
        pred_str = " ".join(tokenizer.convert_ids_to_tokens(collapsed))
        full_phoneme_parts.append(pred_str.strip())

        num_frames     = logits.shape[1]
        frame_duration = WHISPER_FRAME_STRIDE_S

        # ---- CTC target for forced-align (same collapse) ----
        targets_list = collapsed                         # reuse the collapse above
        if not targets_list:
            continue

        log_probs      = torch.log_softmax(logits.float(), dim=-1).cpu().contiguous()
        targets        = torch.tensor([targets_list], dtype=torch.int32)
        input_lengths  = torch.tensor([log_probs.shape[1]], dtype=torch.int32)
        target_lengths = torch.tensor([targets.shape[1]], dtype=torch.int32)
        try:
            aligned, _ = torchaudio.functional.forced_align(
                log_probs, targets, input_lengths, target_lengths, blank=blank_id
            )
        except Exception as e:
            print(f"  [forced_align skipped] chunk {ci} "
                  f"{chunk['start']:.2f}-{chunk['end']:.2f}: {e}")
            continue
        aligned = aligned[0].tolist()

        spans, run_id, run_start = [], None, 0
        for f, tid in enumerate(aligned):
            if tid != run_id:
                if run_id is not None and run_id != blank_id:
                    spans.append((run_id, run_start))
                run_id, run_start = tid, f
        if run_id is not None and run_id != blank_id:
            spans.append((run_id, run_start))

        chunk_aligns = []
        for token_id, sframe in spans:
            phoneme = tokenizer.decode([token_id])
            if token_id == tokenizer.unk_token_id or phoneme in ["", " ", "[", "|"]:
                continue
            if len(phoneme) == 1 and unicodedata.combining(phoneme):
                if chunk_aligns:
                    chunk_aligns[-1]["phoneme"] += phoneme
                continue
            start_time = start_sample / 16000 + sframe * frame_duration
            chunk_aligns.append({"phoneme": phoneme, "start": start_time, "end": None})
        for i in range(len(chunk_aligns) - 1):
            chunk_aligns[i]["end"] = chunk_aligns[i + 1]["start"]
        if chunk_aligns:
            chunk_aligns[-1]["end"] = start_sample / 16000 + num_frames * frame_duration

        all_alignments.extend(chunk_aligns)

    full_phoneme_string = " ".join(full_phoneme_parts).strip()
    return full_phoneme_string, all_alignments
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
def pkl_to_etf(pkl_path, output_path, use_hyp=True):

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    with open(output_path, "w", encoding="utf-8") as out:

        for filename, content in data.items():

            style = content.get("style", "-")

            key = "hyp_intervals" if use_hyp else "ref_intervals"
            intervals = sorted(content[key], key=lambda x: x["start"])

            if not intervals:
                continue

            # 🔥 récupérer tous les phonèmes présents
            phonemes = sorted(set(seg["phoneme"] for seg in intervals))

            # 🔥 timeline complète (tous segments)
            for target_phoneme in phonemes:

                for seg in intervals:
                    start = seg["start"]
                    end = seg["end"]
                    duration = end - start
                    phoneme = seg["phoneme"]

                    if duration <= 0:
                        continue

                    # 🔥 logique clé
                    decision = "t" if phoneme == target_phoneme else "f"

                    line = (
                        f"{filename} 1 "
                        f"{start:.6f} {duration:.6f} "
                        f"sc - {target_phoneme} - {decision}\n"
                    )

                    out.write(line)
VOWELS = set("aeiouyɛøœɔɑɨɪʊʌɒɜəɛ")
foreign_phonemes = {
    'β','θ','ɹ','ɾ','ɣ','ʌ','ʊ','ɪ','ɨ','ɨ̃','ɜ','ɒ','õ','ũ'
}
REF_MAPPING = {
    # Consonants
    "Z": "ʒ","A":"a","S": "ʃ","R": "ʁ","r": "ʁ","N": "ŋ", "J": "ɲ","H": "ɥ","g": "ɡ","Z=": "ʒ","E": "ɛ","O": "ɔ","2": "ø",
"9": "œ","@": "ə","a~": "ɑ̃","o~": "ɔ̃","e~": "ɛ̃","9~": "ɛ̃","E": "ɛ","m=": "m","n=": "n","9~": "ɛ̃",}
NASAL_CANONICAL = {"ã": "ɑ̃", "ẽ": "ɛ̃","ĩ": "ɛ̃","ỹ": "ɛ̃","œ̃": "ɛ̃","ə̃": "ɛ̃",'ø̃':"ɛ̃"}
HYP_PROJECTION = {"ɪ": "i", "ʊ": "u","ɨ": "i", "ɜ": "ə","ʌ": "ɔ","ɒ": "ɔ","ɣ": "ʁ", "ɹ": "ʁ", "ɾ": "ʁ","ʎ": "l",
"β": "b","θ": "t","c": "k","ɑ": "a","mʲ":"m","ɟ": "ɲ","ñ":"ɲ","ṽ":"v"}
asr_to_ipa = {
    "aa": "a",        # a, ɑ
    "bb": "b",
    "kk": "k",        # c, k
    "dd": "d",
    "jj": "dʒ",       # also ʒ (see note below)
    "ei": "e",
    "ff": "f",
    "ii": "i",
    "yy": "j",
    "ll": "l",        # also ʎ
    "mm": "m",        # also mʲ
    "nn": "n",        # also ŋ
    "au": "o",
    "pp": "p",
    "ss": "s",        # also ts
    "tt": "t",
    "ch": "ʃ",        # also tʃ
    "ou": "u",
    "vv": "v",
    "ww": "w",
    "uu": "y",
    "zz": "z",
    "eu": "ø",
    "oe": "œ",
    "an": "ɑ̃",
    "oo": "ɔ",
    "on": "ɔ̃",
    "ee": "ə",
    "ai": "ɛ",
    "in": "ɛ̃",
    "un": "ɛ̃",       # second nasal mapping
    "gn": "ɲ",        # also ɟ depending on system
    "gg": "ɡ",
    "uy": "ɥ",
    "rr": "ʁ",
    "r": "ʁ",
    "SIL": "_"
}
import librosa
def dict_to_csv(data_dict, output_path='phonemes.csv'):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'predicted_phonemes'])
        for filename, info in data_dict.items():
            phonemes = ' '.join([interval['phoneme'] for interval in info['hyp_intervals']])
            writer.writerow([filename, phonemes])
def extract_phones_from_textgrid_typaloc(textgrid_path, remove_silence=True):

    tg = textgrid.openTextgrid(
        textgrid_path,
        includeEmptyIntervals=False,
        duplicateNamesMode="rename"
    )

    # 🔎 Find tier containing "corr"
    tier_name = None
    for t in tg.tierNames:
        if "corr" in t.lower():
            tier_name = t
            break

    # fallback to first tier if none found
    if tier_name is None:
        tier_name = tg.tierNames[0]

    phone_tier = tg.getTier(tier_name)
    
    phones = []
    intervals = []
    
    for start, end, label in phone_tier.entries:
        
        label = label.strip()
        
        # Skip empty intervals
        if label == "":
            continue
        
        # Optionally remove silence
        if remove_silence and label in ["sil", "sp", "spn"]:
            continue
        
        phones.append(label)
        intervals.append((start, end, label))
    
    return phones, intervals
def extract_phones_from_textgrid(tg_path, remove_silence=True,t=""):
    """
    Extract phoneme sequence and timestamps from MFA TextGrid.

    Returns:
        phones: list of phoneme labels
        intervals: list of (start, end, phone)
    """
    
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    
    # List available tiers
   # print("Available tiers:", tg.tierNames)
    
    # Usually MFA phoneme tier is named "phones"
    phone_tier = tg.getTier(t)
    
    phones = []
    intervals = []
    
    for start, end, label in phone_tier.entries:
        
        label = label.strip()
        
        # Skip empty intervals
        if label == "":
            continue
        
        # Optionally remove silence
        if remove_silence and label in ["SIL","sil", "sp", "spn"]:
            continue
        
        phones.append(label)
        intervals.append((start, end, label))
    
    return phones, intervals
def get_reference_alignments(textgrid_path,t=""):

    tg = TextGrid()
    tg.read(textgrid_path)

    ref_alignments = []

    # assuming tier name is "phones"
    tier = tg.getFirst(t)
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
def get_reference_alignments_typaloc(textgrid_path, remove_silence=True):

    tg = textgrid.openTextgrid(
        textgrid_path,
        includeEmptyIntervals=True,
        duplicateNamesMode="rename"
    )

    # 🔎 Find tier containing "corr"
    tier_name = None
    for t in tg.tierNames:
        if "corr" in t.lower():
            tier_name = t
            break

    # fallback to first tier if none found
    if tier_name is None:
        tier_name = tg.tierNames[0]

    tier = tg.getTier(tier_name)

    ref_alignments = []

    for start, end, label in tier.entries:

        label = label.strip()

        # Skip empty intervals
        if not label:
            continue

        # Remove silence if requested
        if remove_silence and label.lower() in {"sil", "sp", "spn"}:
            continue

        ref_alignments.append({
            "phoneme": label,
            "start": start,
            "end": end
        })

    return ref_alignments

def get_ref_intervals_rhap(clean_ref, audio_path):
    """
    Corrects ref timestamps only when TextGrid has a genuine session-level offset
    (i.e. ref extends beyond audio duration by more than threshold seconds).
    No reference to hyp is used — fully independent correction.
    """
    audio, sr = sf.read(audio_path)
    audio_duration = len(audio) / sr
    ref_last = clean_ref[-1]["end"]
    ref_offset = clean_ref[0]["start"]
    
    return [{
            "phoneme": item["phoneme"],
            "start": item["start"] - ref_offset,
            "end": item["end"] - ref_offset
        } for item in clean_ref]
def normalize_phones(seq):
    tokens = seq.strip().split()
    new_tokens = []

    for t in tokens:
        if t == "ɑ":        # only replace standalone ɑ
            new_tokens.append("a")
        if t =='ø̃':
            new_tokens.append('ø')
        else:
            new_tokens.append(t)

    return " ".join(new_tokens)
    

def normalize_phoneme(ph, is_hyp=False):
    if ph is None:
        return None
    
    if ph in {"_"," ","sil","SIL","spn","%","?","0","=","fe~","Ra~","sjo~","~e"}:
        return None
    ph = unicodedata.normalize("NFC", ph)
    if ph and all(unicodedata.combining(c) for c in ph):
        return None

    # cleanup FIRST
    ph = ph.replace(" ", "")
    ph = re.sub(r"@t", "ə", ph)
    ph = re.sub(r"[\?\@]", lambda m: "" if m.group()=="?" else "ə", ph)
    ph = ph.replace("<p:>", "")
    ph = re.sub(r"\u0303+", "\u0303", ph)

    # truncate non-vowel multigraphs BEFORE mapping
    if len(ph) > 1 and ph not in REF_MAPPING and ph[0] not in VOWELS:
        ph = ph[0]

    # now map
    if not is_hyp and ph in REF_MAPPING:
        ph = REF_MAPPING[ph]
    if is_hyp and ph in HYP_PROJECTION:
        ph = HYP_PROJECTION[ph]

    
    if ph in NASAL_CANONICAL:
        ph = NASAL_CANONICAL[ph]
    ph = ph.replace("ː", "")
    if ph in foreign_phonemes:
        return None
    return ph or None

def normalize_phoneme_typaloc(ph):
    if ph is None:
        return None
    # Unicode normalization
    ph = unicodedata.normalize("NFC", ph)
    # enlever contenu [[...]]
    ph = re.sub(r"\[\[.*?\]\]", "", ph)

    # enlever crochets restants mal formés
    ph = re.sub(r"\[\[|\]\]", "", ph)

    # enlever contenu entre parenthèses
    ph = re.sub(r"\(.*?\)", "", ph)

    # enlever NONCORR
    ph = re.sub(r"\bNONCORR\b", "", ph)

    # nettoyer espaces
    ph = re.sub(r"\s+", " ", ph).strip()
    
    
    # enlever espaces multiples
    ph = re.sub(r"\s+", " ", ph).strip()
    ph =re.sub(r"\n.*", "", ph, flags=re.DOTALL)
    ph = ph.replace("yu","uy")
    ph = ph.replace("nn+yy","nn")
    ph = ph.replace("ei\t\t","ei")
    if "NB sur tDeb" in ph:
        ph="ei"
    ph = ph.replace("#a","a")
    ph = ph.replace("kk+","k")
    ph = re.sub(r"\*.*?\*", "", ph)
    ph = re.sub(r"\[\s*pause\s*\]", "", ph)
    ph = re.sub(r"\b\w*pause\w*\b", "", ph)
    
    if ph in ["_", "sil", "spn", "%", "?", "??","0","#", "=","euh","#erreur#"]:
        return None

    # Remove standalone combining marks
    if ph and all(unicodedata.combining(c) for c in ph):
        return None
    # --- Reference mapping ---
    if ph in asr_to_ipa.keys():
        ph = asr_to_ipa[ph]
    ph = ph.replace("dʒ","ʒ")
    return ph
def match_alignments_lev(ref_alignments, hyp_alignments, ref_seq, hyp_seq,
                         max_time_diff=None):  # None = no filtering by default

    alignment = align_sequences(ref_seq, hyp_seq)#align_sequences_banded(ref_seq, hyp_seq, band=band)#

    start_errors = []
    end_errors = []
    duration_errors = []
    mid_errors = []
    matched_pairs = []
    filtered_count = 0

    for ref_idx, hyp_idx in alignment:

        if ref_idx is None or hyp_idx is None:
            continue

        if ref_seq[ref_idx] != hyp_seq[hyp_idx]:
            continue

        r_start = ref_alignments[ref_idx]["start"]
        r_end   = ref_alignments[ref_idx]["end"]
        h_start = hyp_alignments[hyp_idx]["start"]
        h_end   = hyp_alignments[hyp_idx]["end"]

        mid_ref  = (r_start + r_end) / 2
        mid_pred = (h_start + h_end) / 2
        mid_error = abs(mid_ref - mid_pred)

        if max_time_diff is not None and mid_error > max_time_diff:
            filtered_count += 1
            continue

        start_errors.append(abs(r_start - h_start))
        end_errors.append(abs(r_end - h_end))
        duration_errors.append(abs((r_end - r_start) - (h_end - h_start)))
        mid_errors.append(mid_error)
        matched_pairs.append((ref_idx, hyp_idx))

    return start_errors, end_errors, duration_errors, mid_errors, matched_pairs, filtered_count

def get_phoneme_alignments_wavlm_ctcfa(model, feature_extractor, tokenizer, audio_path):
    """
    CTC forced-alignment variant (row 10).
    Same as the naive function except phoneme START frames come from
    torchaudio.functional.forced_align over the full CTC posteriors,
    not from the argmax best-path. Returned phoneme string == naive's,
    so PER is identical and boundary placement is the only variable.
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(audio)

    chunks   = vad_chunk_with_timestamps(audio_path,max_chunk_duration=30)
    device   = next(model.parameters()).device
    blank_id = model.config.pad_token_id

    all_alignments     = []
    full_phoneme_parts = []

    for chunk in chunks:
        start_sample = int(chunk["start"] * 16000)
        end_sample   = int(chunk["end"]   * 16000)
        chunk_tensor = wav[start_sample:end_sample]

        inputs = feature_extractor(
            chunk_tensor.numpy(), sampling_rate=16000, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits                 # (1, T, V)

        predicted_ids = torch.argmax(logits, dim=-1)[0]      # greedy best path

        tokens = tokenizer.convert_ids_to_tokens(predicted_ids.tolist())
        probs = torch.softmax(logits[0], dim=-1)

    
        
        # --- unchanged: decoded string keeps PER identical to the naive system ---
        decoded = tokenizer.batch_decode(predicted_ids.unsqueeze(0))[0]

        full_phoneme_parts.append(decoded.strip())

        num_frames     = logits.shape[1]
        chunk_duration = (end_sample - start_sample) / 16000
        frame_duration = chunk_duration / num_frames

        # ---- CTC target = greedy-collapsed label sequence (repeats collapsed, blanks dropped) ----
        # identical to what the naive decoder emits, so both systems align the SAME phonemes
        targets_list, prev = [], None
        for t in predicted_ids.tolist():
            if t != prev and t != blank_id:
                targets_list.append(t)
            prev = t
        if not targets_list:
            continue                                        # silence / blank-only chunk

        # ---- forced-align that sequence against the full posteriors ----
        log_probs      = torch.log_softmax(logits.float(), dim=-1).cpu().contiguous()
        targets        = torch.tensor([targets_list], dtype=torch.int32)
        input_lengths  = torch.tensor([log_probs.shape[1]], dtype=torch.int32)
        target_lengths = torch.tensor([targets.shape[1]], dtype=torch.int32)
        try:
            aligned, _ = torchaudio.functional.forced_align(
                log_probs, targets, input_lengths, target_lengths, blank=blank_id
            )
        except Exception as e:                              # e.g. T too short for the token count
            print(f"[forced_align skipped] {audio_path} "
                  f"{chunk['start']:.2f}-{chunk['end']:.2f}: {e}")
            continue
        aligned = aligned[0].tolist()                       # token id per frame (blanks = blank_id)

        # ---- merge frames into one span per token instance; keep its onset frame ----
        spans, run_id, run_start = [], None, 0
        for f, tid in enumerate(aligned):
            if tid != run_id:
                if run_id is not None and run_id != blank_id:
                    spans.append((run_id, run_start))
                run_id, run_start = tid, f
        if run_id is not None and run_id != blank_id:
            spans.append((run_id, run_start))

        # ---- spans -> alignments (same skip / combining-mark rules as the naive version) ----
        chunk_aligns = []
        for token_id, sframe in spans:
            phoneme = tokenizer.decode([token_id])
            if token_id == tokenizer.unk_token_id or phoneme in ["", " ", "[","|"]:
                continue
            if len(phoneme) == 1 and unicodedata.combining(phoneme):
                if chunk_aligns:
                    chunk_aligns[-1]["phoneme"] += phoneme
                continue
            start_time = start_sample / 16000 + sframe * frame_duration
            chunk_aligns.append({"phoneme": phoneme, "start": start_time, "end": None})
        # contiguous ends (end_i = start_{i+1}; last = chunk end) — same convention as naive
        for i in range(len(chunk_aligns) - 1):
            chunk_aligns[i]["end"] = chunk_aligns[i + 1]["start"]
        if chunk_aligns:
            chunk_aligns[-1]["end"] = start_sample / 16000 + num_frames * frame_duration

        all_alignments.extend(chunk_aligns)

    full_phoneme_string = " ".join(full_phoneme_parts)
    return full_phoneme_string.strip(), all_alignments
def clean_alignment_dict(alignment_list,flag="",is_hyp=False):
    """
    Normalize phonemes and remove empty or deleted ones.
    Keeps timestamps aligned.
    """
    
    cleaned = []
    
    for item in alignment_list:
        phoneme = item["phoneme"]

        if flag=="typaloc":
            phoneme_norm = normalize_phoneme_typaloc(phoneme)
        else:
            phoneme_norm = normalize_phoneme(phoneme, is_hyp=is_hyp)
        
        # Remove empty phonemes after normalization
        if phoneme_norm == "" or phoneme_norm is None:
            continue
        
        cleaned.append({
            "phoneme": phoneme_norm,
            "start": item["start"],
            "end": item["end"]
        })
    
    return cleaned




def get_phoneme_alignments(model, processor, audio_path):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
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
        end_sample   = int(chunk["end"]   * 16000)
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

        num_frames     = logits.shape[1]
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
def extract_phoneme_sequence(alignment_list):
    return [item["phoneme"] for item in alignment_list]


def get_ref_intervals(clean_ref, audio_path, threshold=0.5):
    audio, sr = sf.read(audio_path)
    audio_duration = len(audio) / sr
    ref_last = clean_ref[-1]["end"]
    ref_offset = clean_ref[0]["start"]

    if ref_last > audio_duration + threshold:
        print(f"  Session offset detected: ref_last={ref_last:.2f}s, "
              f"audio={audio_duration:.2f}s, offset={ref_offset:.3f}s")
        return [{
            "phoneme": item["phoneme"],
            "start": item["start"] - ref_offset,
            "end": item["end"] - ref_offset
        } for item in clean_ref]
    else:
        return list(clean_ref)  # ← this was missing
    


def get_hyp_intervals(clean_hyp):
    """
    Hyp timestamps are always audio-relative (CTC frame * stride).
    No correction needed.
    """
    return list(clean_hyp)


import math, unicodedata, torch, torchaudio, numpy as np, soundfile as sf, librosa

WAVLM_STRIDE_SAMPLES = 320
WAVLM_STRIDE_S       = 0.02
MIN_SAMPLES          = 3200   # pad tiny chunks so the conv stack doesn't choke

def get_phoneme_alignments_w2v_ctcfa(model, processor, audio_path):
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(audio).float()

    chunks   = vad_chunk_with_timestamps(audio_path,max_chunk_duration=30)
    device   = next(model.parameters()).device
    blank_id = model.config.pad_token_id
    unk_id   = processor.tokenizer.unk_token_id

    all_alignments, full_phoneme_parts = [], []

    for chunk in chunks:
        start_sample = int(chunk["start"] * 16000)
        end_sample   = int(chunk["end"]   * 16000)
        chunk_tensor = wav[start_sample:end_sample]
        real_samples = chunk_tensor.shape[-1]
        if real_samples == 0:
            continue
        if real_samples < MIN_SAMPLES:                       # pad short chunks
            chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, MIN_SAMPLES - real_samples))

        inputs = processor(chunk_tensor.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits                  # (1, T, V)

        # keep only frames belonging to the REAL (unpadded) audio
        valid_frames = max(1, min(math.ceil(real_samples / WAVLM_STRIDE_SAMPLES), logits.shape[1]))
        logits = logits[:, :valid_frames, :]

        predicted_ids = torch.argmax(logits, dim=-1)[0]

        # greedy collapse -> FA target + display string
        collapsed, prev = [], None
        for t in predicted_ids.tolist():
            if t != prev and t != blank_id:
                collapsed.append(t)
            prev = t
        full_phoneme_parts.append(" ".join(processor.tokenizer.convert_ids_to_tokens(collapsed)).strip())
        if not collapsed:
            continue

        # CTC forced alignment of the predicted sequence
        log_probs      = torch.log_softmax(logits.float(), dim=-1).cpu().contiguous()
        targets        = torch.tensor([collapsed], dtype=torch.int32)
        input_lengths  = torch.tensor([log_probs.shape[1]], dtype=torch.int32)
        target_lengths = torch.tensor([targets.shape[1]], dtype=torch.int32)
        try:
            aligned, _ = torchaudio.functional.forced_align(
                log_probs, targets, input_lengths, target_lengths, blank=blank_id)
        except Exception as e:
            print(f"  [forced_align skipped] {chunk['start']:.2f}-{chunk['end']:.2f}: {e}")
            continue
        aligned = aligned[0].tolist()

        # runs with start AND end frame (silence between phonemes stays unassigned)
        spans, run_id, run_start = [], None, 0
        for fi, tid in enumerate(aligned):
            if tid != run_id:
                if run_id is not None and run_id != blank_id:
                    spans.append((run_id, run_start, fi))
                run_id, run_start = tid, fi
        if run_id is not None and run_id != blank_id:
            spans.append((run_id, run_start, len(aligned)))

        off = start_sample / 16000
        chunk_aligns = []
        for token_id, sframe, eframe in spans:
            phoneme = processor.decode([token_id])
            if token_id == unk_id or phoneme in ["", " ", "[", "|"]:
                continue
            if len(phoneme) == 1 and unicodedata.combining(phoneme):
                if chunk_aligns:
                    chunk_aligns[-1]["phoneme"] += phoneme
                    chunk_aligns[-1]["end"] = off + eframe * WAVLM_STRIDE_S
                continue
            chunk_aligns.append({"phoneme": phoneme,
                                 "start": off + sframe * WAVLM_STRIDE_S,
                                 "end":   off + eframe * WAVLM_STRIDE_S})
        all_alignments.extend(chunk_aligns)

    return " ".join(full_phoneme_parts).strip(), all_alignments

"""
WhisperX-style VAD chunking for the Whisper-encoder + CTC phoneme model.

Built on your approach: uses whisperx.vads.pyannote.load_vad_model, reads the
RAW per-frame VAD scores, and binarizes them with WhisperX's onset/offset
thresholds. The goal is unchanged from the original vad_chunk_with_timestamps:
return a list of {"start", "end"} (seconds, original time) chunks, each <= 30 s,
ready for the inference loop.

Two stages:
  1. VAD + binarize  -> WhisperX raw scores -> speech segments (hysteresis: go
                        active above `onset`, inactive below `offset`).
  2. cut & merge     -> pack segments into <= chunk_size windows, KEEPING internal
                        pauses inside a window. A single segment longer than
                        chunk_size is split at its QUIETEST frame (real WhisperX
                        behaviour, possible here because we kept the raw scores),
                        so nothing ever hits Whisper's 30 s truncation.

Only load_whisperx_vad() touches whisperx, so the rest is importable/testable
offline without it.
"""

import numpy as np
import torch
from whisperx.vads.pyannote import load_vad_model
# WhisperX defaults
ONSET = 0.5
OFFSET = 0.363


def load_whisperx_vad(wav):
    vad_pipeline = load_vad_model(
    device="cuda",
    token=os.environ["HF_TOKEN"])
    vad_scores = vad_pipeline(wav)
    scores = vad_scores.data[:, 0]
    frames = vad_scores.sliding_window
    times = [frames[i].middle for i in range(len(scores))]
    return scores, times

def _binarize(scores, times, onset=ONSET, offset=OFFSET):
    """Hysteresis binarization -> list of (start_s, end_s) speech segments."""
    segments = []
    is_active = scores[0] > onset
    start = times[0] if is_active else None
    for t, sc in zip(times[1:], scores[1:]):
        if is_active:
            if sc < offset:
                segments.append((start, t))
                is_active = False
        else:
            if sc > onset:
                start = t
                is_active = True
    if is_active:
        segments.append((start, times[-1]))
    return segments


def _split_long_by_score(seg_start, seg_end, times, scores, chunk_size):
    """Split a > chunk_size segment at the quietest frame in each window.

    Mirrors WhisperX: when a segment exceeds max_duration, cut at the lowest
    detection score in the second half rather than at a hard time boundary, so
    the cut lands on minimally-active speech.
    """
    pieces, cur = [], seg_start
    while seg_end - cur > chunk_size:
        lo, hi = cur + chunk_size * 0.5, cur + chunk_size
        idx = np.where((times >= lo) & (times <= hi))[0]
        cut = times[idx[np.argmin(scores[idx])]] if len(idx) else cur + chunk_size
        pieces.append((cur, cut))
        cur = cut
    pieces.append((cur, seg_end))
    return pieces


def merge_chunks(segments, times, scores, chunk_size=20.0):
    """WhisperX cut & merge -> list of {"start","end"} chunks, each <= chunk_size."""
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    split = []
    for s, e in segments:
        if e - s > chunk_size:
            split.extend(_split_long_by_score(s, e, times, scores, chunk_size))
        else:
            split.append((s, e))

    if not split:
        return []

    merged = []
    curr_start, curr_end = split[0]
    for s, e in split[1:]:
        if e - curr_start > chunk_size and curr_end - curr_start > 0:
            merged.append({"start": curr_start, "end": curr_end})
            curr_start = s
        curr_end = e
    merged.append({"start": curr_start, "end": curr_end})
    return merged


def vad_chunk_with_timestamps(
    wav,
    sampling_rate=16000,
    max_chunk_duration=8.0,
    onset=ONSET,
    offset=OFFSET,
):
    """Drop-in replacement. Same return type as the old rVAD version.

    wav               : torch.Tensor (1D, 16 kHz)  -- or a file path
    vad_model         : object from load_whisperx_vad() (load it once, reuse it)
    max_chunk_duration: keep <= 30.0 for Whisper's hard cap
    """
    scores, times = load_whisperx_vad(wav)
    segments = _binarize(scores, times, onset, offset)
    return merge_chunks(segments, times, scores, chunk_size=max_chunk_duration)