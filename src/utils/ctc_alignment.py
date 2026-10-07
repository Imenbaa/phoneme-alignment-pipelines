"""Phoneme alignment from a CTC acoustic model.

One implementation covers every CTC front-end in the benchmark; the differences
between wav2vec 2.0 / WavLM / HuBERT and a Whisper encoder are expressed as
arguments, not as separate copies:

    decoder="forced_align"  torchaudio forced alignment of the greedy-collapsed
                            label sequence against the full posteriors
    decoder="greedy"        boundaries read straight off the argmax best path
    end_mode="span"         a phoneme ends where its own frame run ends, so the
                            silence between phonemes stays unassigned
    end_mode="contiguous"   a phoneme ends where the next one starts

The decoded phoneme string is identical under both decoders (both emit the
greedy-collapsed sequence), so PER is unaffected and boundary placement is the
only variable -- which is what makes the two comparable.

Chunking is injected: `chunker` is any callable taking the audio path and
returning ``[{"start", "end"}, ...]`` in seconds. The default runs the whole
file in one pass, so nothing here depends on a VAD being installed; pass
:func:`make_vad_chunker` to use the project's WhisperX-style VAD.
"""

import math
import unicodedata

import torch

from .audio_io import DEFAULT_SAMPLE_RATE, load_audio_mono

#: 20 ms at 16 kHz -- the wav2vec2/WavLM conv stack, and the Whisper encoder
#: (10 ms mel hop, halved again by the stride-2 convolution).
CTC_FRAME_STRIDE_SAMPLES = 320
CTC_FRAME_STRIDE_S = 0.02

#: Tokens that are never phonemes, whatever the vocabulary.
SKIP_TOKENS = ("", " ", "[", "|")

#: Pad chunks shorter than this so the conv front-end does not choke.
MIN_CHUNK_SAMPLES = 3200


# ---------------------------------------------------------------------------
# CTC primitives
# ---------------------------------------------------------------------------
def greedy_collapse(token_ids, blank_id):
    """CTC collapse: drop repeats, then drop blanks."""
    collapsed, prev = [], None
    for token_id in token_ids:
        if token_id != prev and token_id != blank_id:
            collapsed.append(token_id)
        prev = token_id
    return collapsed


def join_tokens(tokens):
    """Join vocabulary tokens into a display string, folding combining marks.

    A tokenizer emits a diacritic as its own token, so a naive space-join yields
    ``"ɑ ̃"`` where the intervals carry ``"ɑ̃"``. Folding keeps the predicted
    string consistent with the interval labels -- which matters when that string
    is written out as the transcript for a downstream aligner.
    """
    joined = []
    for token in tokens:
        if joined and token and all(unicodedata.combining(c) for c in token):
            joined[-1] += token
        else:
            joined.append(token)
    return " ".join(joined).strip()


def forced_align_tokens(logits, targets_list, blank_id):
    """Forced-align `targets_list` against the posteriors; one token per frame."""
    import torchaudio

    log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().contiguous()
    targets = torch.tensor([targets_list], dtype=torch.int32)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.int32)
    target_lengths = torch.tensor([targets.shape[1]], dtype=torch.int32)
    aligned, _ = torchaudio.functional.forced_align(
        log_probs, targets, input_lengths, target_lengths, blank=blank_id
    )
    return aligned[0].tolist()


def token_runs(frame_tokens, blank_id):
    """Merge consecutive identical frames into ``(token_id, start, end)`` runs.

    Blank runs are dropped; `end` is exclusive.
    """
    runs, run_id, run_start = [], None, 0
    for frame, token_id in enumerate(frame_tokens):
        if token_id != run_id:
            if run_id is not None and run_id != blank_id:
                runs.append((run_id, run_start, frame))
            run_id, run_start = token_id, frame
    if run_id is not None and run_id != blank_id:
        runs.append((run_id, run_start, len(frame_tokens)))
    return runs


def runs_to_intervals(
    runs,
    decode,
    offset_s,
    frame_duration,
    num_frames,
    unk_id=None,
    end_mode="span",
    skip_tokens=SKIP_TOKENS,
):
    """Turn frame runs into ``[{"phoneme", "start", "end"}, ...]`` in seconds.

    Combining marks (diacritics emitted as their own token) are appended to the
    preceding phoneme rather than becoming intervals of their own.
    """
    intervals = []
    for token_id, start_frame, end_frame in runs:
        phoneme = decode(token_id)
        if token_id == unk_id or phoneme in skip_tokens:
            continue
        if len(phoneme) == 1 and unicodedata.combining(phoneme):
            if intervals:
                intervals[-1]["phoneme"] += phoneme
                if end_mode == "span":
                    intervals[-1]["end"] = offset_s + end_frame * frame_duration
            continue
        intervals.append(
            {
                "phoneme": phoneme,
                "start": offset_s + start_frame * frame_duration,
                "end": offset_s + end_frame * frame_duration,
            }
        )

    if end_mode == "contiguous":
        for i in range(len(intervals) - 1):
            intervals[i]["end"] = intervals[i + 1]["start"]
        if intervals:
            intervals[-1]["end"] = offset_s + num_frames * frame_duration
    return intervals


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def whole_file_chunker(audio_path, sample_rate=DEFAULT_SAMPLE_RATE):
    """Single chunk covering the whole file -- the default, dependency-free."""
    from .audio_io import audio_duration

    return [{"start": 0.0, "end": audio_duration(audio_path)}]


def fixed_window_chunker(window_s=30.0):
    """Chunker cutting the file into fixed windows, with no VAD."""

    def chunker(audio_path, sample_rate=DEFAULT_SAMPLE_RATE):
        from .audio_io import audio_duration

        duration = audio_duration(audio_path)
        chunks, start = [], 0.0
        while start < duration:
            chunks.append({"start": start, "end": min(start + window_s, duration)})
            start += window_s
        return chunks

    return chunker


def make_vad_chunker(max_chunk_duration=30.0, **kwargs):
    """Chunker backed by the project's WhisperX-style VAD (optional dependency)."""

    def chunker(audio_path, sample_rate=DEFAULT_SAMPLE_RATE):
        from .VAD_chunk import vad_chunk_with_timestamps

        return vad_chunk_with_timestamps(
            audio_path, max_chunk_duration=max_chunk_duration, **kwargs
        )

    return chunker


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def ctc_align_audio(
    model,
    audio_path,
    processor=None,
    feature_extractor=None,
    tokenizer=None,
    chunker=None,
    blank_id=None,
    frame_stride_s=CTC_FRAME_STRIDE_S,
    frame_stride_samples=CTC_FRAME_STRIDE_SAMPLES,
    trim_padding_frames=True,
    min_samples=0,
    decoder="forced_align",
    end_mode="span",
    sample_rate=DEFAULT_SAMPLE_RATE,
    skip_tokens=SKIP_TOKENS,
    verbose=False,
):
    """Align one audio file with a CTC phoneme model.

    Args:
        model: a CTC model whose ``logits`` are ``(1, T, V)``.
        processor: a combined processor, or give `feature_extractor` and
            `tokenizer` separately.
        chunker: ``callable(audio_path, sample_rate) -> [{"start","end"}]``.
            Defaults to :func:`whole_file_chunker`.
        blank_id: CTC blank; taken from ``model.config.pad_token_id`` if omitted.
        frame_stride_s: seconds per output frame. Pass ``None`` to derive it
            from chunk duration / frame count instead of assuming a stride.
        trim_padding_frames: drop the frames that correspond to padding, needed
            for encoders with a fixed output length (Whisper) and for padded
            short chunks.
        min_samples: pad chunks shorter than this many samples.
        decoder: ``"forced_align"`` or ``"greedy"``.
        end_mode: ``"span"`` or ``"contiguous"`` (see the module docstring).

    Returns:
        ``(phoneme_string, intervals)`` -- the greedy-collapsed phoneme string
        and the interval list, in seconds, over the whole file.
    """
    if decoder not in ("forced_align", "greedy"):
        raise ValueError(f"decoder must be 'forced_align' or 'greedy', got {decoder!r}")
    if end_mode not in ("span", "contiguous"):
        raise ValueError(f"end_mode must be 'span' or 'contiguous', got {end_mode!r}")

    featurizer, tok = _resolve_io(processor, feature_extractor, tokenizer)

    audio, sample_rate = load_audio_mono(audio_path, target_sr=sample_rate)
    wav = torch.from_numpy(audio).float()

    chunks = (chunker or whole_file_chunker)(audio_path, sample_rate)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if blank_id is None:
        blank_id = getattr(model.config, "pad_token_id", None)
    if blank_id is None:
        blank_id = tok.pad_token_id
    unk_id = getattr(tok, "unk_token_id", None)

    all_intervals, phoneme_parts = [], []

    for chunk in chunks:
        start_sample = int(chunk["start"] * sample_rate)
        end_sample = int(chunk["end"] * sample_rate)
        chunk_tensor = wav[start_sample:end_sample]
        real_samples = chunk_tensor.shape[-1]
        if real_samples == 0:
            continue
        if real_samples < min_samples:
            chunk_tensor = torch.nn.functional.pad(
                chunk_tensor, (0, min_samples - real_samples)
            )

        inputs = featurizer(
            chunk_tensor.numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {
            k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            logits = model(**inputs).logits  # (1, T, V)

        if trim_padding_frames and frame_stride_samples:
            valid = max(1, math.ceil(real_samples / frame_stride_samples))
            logits = logits[:, : min(valid, logits.shape[1]), :]

        num_frames = logits.shape[1]
        # Derived frame duration uses the chunk's nominal length, as in the paper.
        # A VAD chunk can end past the audio, which stretches its frames slightly.
        frame_duration = (
            frame_stride_s
            if frame_stride_s is not None
            else ((end_sample - start_sample) / sample_rate) / num_frames
        )

        predicted_ids = torch.argmax(logits, dim=-1)[0]
        collapsed = greedy_collapse(predicted_ids.tolist(), blank_id)
        phoneme_parts.append(join_tokens(tok.convert_ids_to_tokens(collapsed)))
        if not collapsed:
            continue

        if decoder == "greedy":
            frame_tokens = predicted_ids.tolist()
        else:
            try:
                frame_tokens = forced_align_tokens(logits, collapsed, blank_id)
            except Exception as exc:  # T too short for the token count, etc.
                if verbose:
                    print(
                        f"  [forced_align skipped] {audio_path} "
                        f"{chunk['start']:.2f}-{chunk['end']:.2f}: {exc}"
                    )
                continue

        all_intervals.extend(
            runs_to_intervals(
                token_runs(frame_tokens, blank_id),
                decode=lambda token_id: tok.decode([token_id]),
                offset_s=start_sample / sample_rate,
                frame_duration=frame_duration,
                num_frames=num_frames,
                unk_id=unk_id,
                end_mode=end_mode,
                skip_tokens=skip_tokens,
            )
        )

    return " ".join(phoneme_parts).strip(), all_intervals


def _resolve_io(processor, feature_extractor, tokenizer):
    """Accept either a combined processor or a feature extractor + tokenizer."""
    featurizer = feature_extractor if feature_extractor is not None else processor
    if featurizer is None:
        raise ValueError("pass processor, or feature_extractor and tokenizer")

    tok = tokenizer
    if tok is None:
        tok = getattr(processor, "tokenizer", None)
    if tok is None:
        tok = getattr(feature_extractor, "tokenizer", None)
    if tok is None:
        raise ValueError("no tokenizer: pass tokenizer= explicitly")
    return featurizer, tok


# ---------------------------------------------------------------------------
# Presets for the encoders used in the benchmark
# ---------------------------------------------------------------------------
def align_wav2vec2_ctc(model, processor, audio_path, **kwargs):
    """wav2vec 2.0 / HuBERT + CTC head, forced alignment, true spans."""
    kwargs.setdefault("min_samples", MIN_CHUNK_SAMPLES)
    kwargs.setdefault("end_mode", "span")
    return ctc_align_audio(model, audio_path, processor=processor, **kwargs)


def align_wavlm_ctc(model, feature_extractor, tokenizer, audio_path, **kwargs):
    """WavLM + CTC head, forced alignment, true spans. As in the paper, the frame
    duration is the chunk duration divided by the frame count, with no padding
    and no frame trimming."""
    kwargs.setdefault("end_mode", "span")
    kwargs.setdefault("frame_stride_s", None)
    kwargs.setdefault("trim_padding_frames", False)
    return ctc_align_audio(
        model,
        audio_path,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        **kwargs,
    )


def align_whisper_encoder_ctc(
    model, feature_extractor, tokenizer, audio_path, **kwargs
):
    """Whisper encoder + CTC head, forced alignment, true spans. The encoder
    always emits 1500 frames, so the padding frames are trimmed back to the real
    audio length before alignment."""
    kwargs.setdefault("end_mode", "span")
    return ctc_align_audio(
        model,
        audio_path,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        **kwargs,
    )


def align_greedy_ctc(model, processor, audio_path, **kwargs):
    """Boundaries from the argmax best path -- the naive baseline."""
    kwargs.setdefault("decoder", "greedy")
    kwargs.setdefault("end_mode", "contiguous")
    return ctc_align_audio(model, audio_path, processor=processor, **kwargs)
