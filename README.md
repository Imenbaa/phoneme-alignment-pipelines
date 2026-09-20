# Comparing Phoneme Alignment Pipelines Across Spontaneous and Pathological French Speech

Code for the SLT 2026 submission benchmarking **five phoneme alignment pipelines** on
French spontaneous and pathological speech.

Phoneme alignment locates phoneme boundaries in speech and is a key step in many speech
processing applications, yet its robustness is under-evaluated for spontaneous and
pathological speech. This work compares **transcription front-ends** (word-level ASR vs.
three phoneme recognizers) against **alignment back-ends** (Montreal Forced Aligner vs.
CTC forced alignment).

**Main findings.** Direct phoneme recognition consistently improves alignment over the
conventional ASR+MFA pipeline, while MFA yields more accurate boundaries than CTC across
all conditions. A phoneme-level analysis shows CTC is most affected by fricatives and
plosives, whereas MFA stays accurate for most phonemes, with increased variability mainly
for glides and some consonants.

## Pipelines evaluated

Each pipeline is a *transcription front-end* → *alignment back-end* pair:

| System | Front-end | Back-end |
|---|---|---|
| `ASR+G2P->MFA` | Word-level ASR, then grapheme-to-phoneme | MFA |
| `w2v->MFA` / `w2v->CTC` | wav2vec 2.0 phoneme recognizer | MFA / CTC |
| `wavlm->MFA` / `wavlm->CTC` | WavLM phoneme recognizer | MFA / CTC |
| `whisper->MFA` / `whisper->CTC` | Whisper phoneme recognizer | MFA / CTC |
| `GoldPh->MFA` | Reference phoneme transcription (topline) | MFA |
| `GoldW+G2P->MFA` | Reference word transcription + G2P (topline) | MFA |

## Corpora

Three French corpora, reported as six condition labels:

| Label | Corpus / group | Style | Files | Phonemes | Task | Accent |
|---|---|---|---|---|---|---|
| `mon` | MonPaGe | Semi | 68 | 30,592 | Picture description | Belgian Fr. |
| `rhap` | Rhapsodie | Spont / Planned / Semi | 38 / 11 / 4 | 46,184 / 29,234 / 14,697 | Interaction | Native Fr. |
| `park` | Typaloc — Parkinson's disease | Read | 8 | 4,571 | Read | South / Paris |
| `cereb` | Typaloc — cerebellar ataxia | Read | 7 | 4,642 | Read | South / Paris |
| `sla` | Typaloc — ALS (SLA) | Read | 12 | 6,396 | Read | South / Paris |
| `ctrl` | Typaloc — healthy controls | Read | 12 | 6,829 | Read | South / Paris |

Rhapsodie is a corpus of contemporary spoken French; 53 of the original 57 recordings are
used (4 excluded for insufficient quality). For all three corpora, annotations and
alignments were produced automatically and then manually verified.

**No speech data, transcriptions, or alignments are included in this repository.** The
corpora contain clinical recordings and are not redistributable; obtain them from their
respective providers. Only aggregate metrics and figures are published here.

## Repository layout

```
src/
  evaluate_alignment.py   evaluation entry point (CLI) -- see "Reproducing" below
  utils/
    ctc_alignment.py        CTC forced alignment / greedy decoding -> phoneme intervals
    whisper_ctc_model.py    Whisper encoder + CTC head
    textgrid_io.py          read a phone tier out of a TextGrid
    phoneme_normalization.py  declarative label -> phoneme normalisation
    phoneme_mappings.py     optional inventory presets (SAMPA/ASR codes -> IPA)
    intervals.py            interval helpers (offsets, sequences)
    corpus.py               pair audio with its annotation by stem
    alignment_matching.py   Levenshtein matching of reference to hypothesis
    metrics_alignment.py    F1@20/50ms, AAS, median boundary error, PER
    export.py               CSV / ETF / pickle dumps
    VAD_chunk.py            WhisperX-style VAD chunking for long recordings
    README.md               module map and the protocol in code
  finetuning/             phoneme recognizer fine-tuning (WavLM, Whisper)
    train_wavlm.py, train_whisper.py, prep_whisper_dataset.py
```

This repository holds the **evaluation and fine-tuning code**. No corpus, checkpoint,
alignment dump or result table is included. The code is corpus-agnostic: the naming
scheme, tier names, phoneme inventory and grouping metadata of the data you point it at
are all command-line arguments, so the evaluation protocol below can be reproduced on any
corpus with reference phoneme segmentation.

## Reproducing

The pipeline runs in **separate conda environments** -- MFA pins its own numpy/scipy and
ships Kaldi binaries, so it cannot share an env with the phoneme-recognition stack.

```bash
# 1. Phoneme recognition + evaluation  (Python 3.10)
conda create -n viterbi python=3.10 && conda activate viterbi
pip install -r requirements.txt

# 2. MFA back-end  (own env; version used for the paper)
conda create -n mfa_env -c conda-forge montreal-forced-aligner=3.3.9

# 3. Optional: VAD / diarization front-end
conda create -n pyannote_env python=3.10 && conda activate pyannote_env
pip install -r requirements-vad.txt
```

Gated Hugging Face models (pyannote, WhisperX VAD) need a token in the environment --
the code reads `HF_TOKEN` and no credentials are stored in this repository:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

### What the evaluation expects

Two directories, paired by filename stem:

```
audio/        utt001.wav  utt002.wav  ...
reference/    utt001.TextGrid  utt002.TextGrid  ...   (a tier of phone intervals)
```

Stems need not match exactly -- `--ref-stem-suffix=-Pro` strips a trailing marker before
pairing. Optionally, a CSV assigns each file a group (speaking style, speaker group,
clinical condition), and metrics are then reported per group as well as globally.

### Scoring a CTC back-end

```bash
python src/evaluate_alignment.py \
    --audio-dir  data/audio \
    --ref-dir    data/reference \
    --ref-tier   phones \
    --hypothesis ctc \
    --model-type wav2vec2 \
    --checkpoint models/wav2vec2-french-phonemizer \
    --chunking   vad \
    --out-dir    results/w2v2_ctc --per-phoneme
```

`--model-type` selects `wav2vec2`, `wavlm` or `whisper`; `--decoder greedy` reads
boundaries off the argmax best path instead of forced-aligning, and decodes the same
phoneme string, so PER is unchanged and boundary placement is the only variable.

### Scoring an MFA back-end

MFA is run outside this repository; point the script at its output TextGrids:

```bash
mfa align <corpus_dir> <phoneme_dict.txt> french_mfa <output_dir> \
    --beam 100 --retry_beam 100

python src/evaluate_alignment.py \
    --audio-dir  data/audio \
    --ref-dir    data/reference --ref-tier phones \
    --hypothesis textgrid --hyp-dir <output_dir> --hyp-tier phones \
    --out-dir    results/mfa --per-phoneme
```

This path imports no torch, so MFA output can be scored without a deep-learning stack.

### Matching label conventions

Reference and hypothesis rarely use the same symbols. `--ref-preset` / `--hyp-preset`
apply a built-in mapping (`french-sampa`, `french-asr-codes`, `french-broad`), and
`--ref-mapping` / `--hyp-mapping` take a JSON object `{"label": "phoneme"}` of your own.
The default is `none`: no mapping is applied unless you ask for one.

Every run writes `inventory_<tag>.txt` comparing the phoneme inventories actually seen on
each side. **Read it first** -- a small shared inventory means the two sides disagree on
notation, which silently depresses PER and boundary recall.

### Output

| File | Contents |
|---|---|
| `metrics_<tag>.csv` | per-file, per-group and global metrics |
| `per_phoneme_<tag>.csv` | per-phoneme breakdown (`--per-phoneme`) |
| `inventory_<tag>.txt` | reference vs. hypothesis inventory comparison |
| `predictions_<tag>.csv` | the predicted phoneme sequence per file |
| `alignment_<tag>.pkl` | the full interval store, for re-scoring without re-running |
| `hyp_<tag>.etf` | hypothesis intervals in ETF form (`--etf`) |

## Metrics

- **F1@20ms / F1@50ms** — boundary detection F1 at 20 ms and 50 ms tolerance
- **MedianBE** — median boundary error (ms)
- **%>50ms** — proportion of boundaries off by more than 50 ms
- **AAS** — average absolute shift (ms)
- **DurErr** — phoneme duration error (ms)
- **onset_bias / offset_bias** — signed boundary bias (ms)
- **PER** — phoneme error rate of the transcription front-end (%)

## Status and caveats

- The evaluation code is corpus-agnostic, but it was written for and validated on the
  three French corpora above. Nothing stops it running elsewhere; nothing guarantees the
  built-in label presets fit another language.
- Boundary resolution is bounded by the encoder frame rate: one frame per 20 ms. Errors
  below that are not measurable by this protocol.
- `--chunking vad` needs `whisperx` and a gated Hugging Face model. The default
  (`none`, whole file) and `fixed` need neither.
- Reference offsets: some annotation carries session-level timestamps that run past the
  audio file. `--ref-offset auto` (the default) re-anchors only when the reference ends
  more than `--ref-offset-tolerance` seconds beyond the audio.

## Acknowledgements

The evaluation code in `src/` was refactored with the help of Claude AI: the research
notebooks and helper scripts used for the paper were extracted into the dataset-independent
modules and command-line entry point documented above, so the evaluation protocol can be
reproduced without the private corpora. The experiments, results and analysis are the
authors' own.

## Citation

Paper under review at SLT 2026. Citation details will be added on acceptance.

```bibtex
@inproceedings{phoneme_alignment_slt2026,
  title     = {Comparing Phoneme Alignment Pipelines Across Spontaneous and
               Pathological French Speech},
  author    = {TODO},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```
