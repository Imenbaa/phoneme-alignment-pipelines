# `src/utils` — evaluation building blocks

Extracted from the old `utils_phoneme_reco.py` and made corpus-agnostic: no
function refers to a specific dataset, and every convention that used to be
hard-coded (tier names, phoneme mappings, silence labels, chunking) is now an
argument with a neutral default.

| Module | What it holds |
| --- | --- |
| `audio_io.py` | `load_audio_mono`, `audio_duration` |
| `phoneme_mappings.py` | Optional inventory presets (SAMPA→IPA, ASR codes→IPA, nasal canonicalisation, narrow→broad projection). Nothing is applied implicitly. |
| `phoneme_normalization.py` | `PhonemeNormalizer` (declarative label→phoneme rules), `clean_alignment_dict`, `normalize_phoneme_sequence`, `french_ipa_normalizer` as a worked example |
| `textgrid_io.py` | `read_phone_intervals`, `extract_phones_from_textgrid`, `select_tier_name`, `list_tier_names` |
| `intervals.py` | `extract_phoneme_sequence`, `shift_intervals`, `correct_interval_offset`, `total_duration` |
| `corpus.py` | `discover_pairs`, `load_group_map`, `assign_groups` — pair audio with annotation by stem |
| `ctc_alignment.py` | CTC primitives, the `ctc_align_audio` driver, chunkers, and per-encoder presets |
| `whisper_ctc_model.py` | `WhisperEncoderForCTC` |
| `alignment_matching.py` | `match_alignments_lev` |
| `metrics_alignment.py` | Metric aggregation (unchanged) |
| `export.py` | `dict_to_csv`, `write_etf`, `pkl_to_etf` |
| `VAD_chunk.py` | WhisperX-style VAD chunking (optional dependency) |

For a ready-made command-line front-end over all of this, see
[`src/evaluate_alignment.py`](../evaluate_alignment.py) and the *Reproducing*
section of the top-level README. The rest of this file is the same protocol in
code, for when you want to drive it yourself.

## The protocol

Every function speaks the same interval format, in **seconds**:

```python
[{"phoneme": str, "start": float, "end": float}, ...]
```

```python
from utils.ctc_alignment import align_wav2vec2_ctc, make_vad_chunker
from utils.intervals import correct_interval_offset, extract_phoneme_sequence
from utils.metrics_alignment import compute_metrics
from utils.phoneme_normalization import PhonemeNormalizer, clean_alignment_dict
from utils.textgrid_io import read_phone_intervals

# 1. Describe your two label conventions once.
ref_norm = PhonemeNormalizer(mapping=MY_REF_TABLE)     # or french_ipa_normalizer("reference")
hyp_norm = PhonemeNormalizer(mapping=MY_MODEL_TABLE)   # or french_ipa_normalizer("hypothesis")

store = {}
for file_id, wav, textgrid in corpus:
    # 2. Reference: read the phone tier, normalise, re-anchor if needed.
    ref = clean_alignment_dict(read_phone_intervals(textgrid, contains="phones"), ref_norm)
    ref = correct_interval_offset(ref, audio_path=wav)

    # 3. Hypothesis: CTC forced alignment, VAD-chunked.
    _, hyp = align_wav2vec2_ctc(model, processor, wav, chunker=make_vad_chunker(30.0))
    hyp = clean_alignment_dict(hyp, hyp_norm)

    store[file_id] = {
        "ref_intervals": ref, "hyp_intervals": hyp,
        "ref_seq": extract_phoneme_sequence(ref),
        "hyp_seq": extract_phoneme_sequence(hyp),
        "style": "ALL",          # any grouping key you want reported separately
    }

# 4. Pooled boundary metrics + PER.
compute_metrics(store, "metrics.csv", per_phoneme_csv="per_phoneme.csv")
```

Notes:

- **Chunking is injected.** `ctc_align_audio` runs the whole file in one pass by
  default, so it imports no VAD. Pass `make_vad_chunker(...)` (needs `whisperx`)
  or `fixed_window_chunker(30.0)` to change that.
- **Decoder vs. boundaries.** `decoder="forced_align"` and `decoder="greedy"`
  emit the *same* phoneme string, so PER is identical and boundary placement is
  the only variable between them. `end_mode="span"` leaves inter-phoneme silence
  unassigned; `end_mode="contiguous"` makes each phoneme end where the next
  begins.
- **Normalisation is explicit.** A bare `PhonemeNormalizer()` only strips
  annotation markup and silence labels. Mappings must be passed in.
