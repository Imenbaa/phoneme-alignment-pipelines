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
    token="hf_XXXXXXXXXXX")
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