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
  utils/                analysis + metric computation (core of the paper)
    w2vctc_rhapsodie.ipynb    MAIN notebook: produces every figure and table
    metrics.py                F1@20/50ms, AAS, median boundary error, duration error
    metrics_alignment.py      alignment-specific metrics
    align_metrics.py          TrackEval-based scoring
    analyze_phonemes.py       per-phoneme / manner-of-articulation breakdown
    utils_phoneme_reco.py     phoneme recognition inference helpers
    prepare_mfa.py            build MFA corpus dirs + dictionaries
    VAD_chunk.py, apply_vad.py    VAD-based chunking for long recordings
    rhapsodie.ipynb, wavlm_rhapsodie.ipynb, whisper_rhapsodie.ipynb,
    typaloc.ipynb, monpage.ipynb, w2v_mfa.ipynb   per-corpus / per-model runs
  finetuning/           phoneme recognizer fine-tuning (WavLM, Whisper)
    train_wavlm.py, train_whisper.py, prep_whisper_dataset.py
  ASR_mfa.ipynb         ASR front-end transcription for the ASR+G2P->MFA baseline
```

This repository holds the **transcription and analysis code**. The CTC alignment
back-end, the MFA configuration, and the aggregate result tables and figures are not
included here.

## Reproducing

The pipeline runs in **separate conda environments** — MFA pins its own numpy/scipy and
ships Kaldi binaries, so it cannot share an env with the phoneme-recognition stack.

```bash
# 1. Phoneme recognition + analysis  (Python 3.10)
conda create -n viterbi python=3.10 && conda activate viterbi
pip install -r requirements.txt

# 2. MFA back-end  (own env; version used for the paper)
conda create -n mfa_env -c conda-forge montreal-forced-aligner=3.3.9

# 3. Optional: VAD / diarization front-end
conda create -n pyannote_env python=3.10 && conda activate pyannote_env
pip install -r requirements-vad.txt
```

Gated Hugging Face models (pyannote, WhisperX VAD) need a token in the environment —
the code reads `HF_TOKEN` and no credentials are stored in this repository:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

### Pipeline steps

1. **Transcription** — run a phoneme recognizer (`src/utils/utils_phoneme_reco.py`, or the
   per-model notebooks) or word-level ASR (`src/ASR_mfa.ipynb`) to produce phoneme/word
   transcriptions per corpus.
2. **CTC back-end** — Viterbi forced alignment over the CTC posteriors (e.g.
   `torchaudio.functional.forced_align`), writing alignments to `ctc_results/`. The
   encoder emits one frame per 20 ms, which bounds boundary resolution.
3. **MFA back-end** — `src/utils/prepare_mfa.py` builds the corpus directory and
   dictionary, then align with MFA 3.3.9 and the `french_mfa` acoustic model:
   ```bash
   mfa align <corpus_dir> <phoneme_dict.txt> french_mfa <output_dir> \
       --beam 100 --retry_beam 100
   ```
   Results land in `mfa_results*/`.
4. **Scoring** — open `src/utils/w2vctc_rhapsodie.ipynb`. It loads the alignment dumps
   from both back-ends, computes the metrics below, and renders the figures.

Steps 1–3 write into `ctc_results/`, `mfa_results*/` and `data/`, which are gitignored:
the notebook expects those directories to exist locally.

## Metrics

- **F1@20ms / F1@50ms** — boundary detection F1 at 20 ms and 50 ms tolerance
- **MedianBE** — median boundary error (ms)
- **%>50ms** — proportion of boundaries off by more than 50 ms
- **AAS** — average absolute shift (ms)
- **DurErr** — phoneme duration error (ms)
- **onset_bias / offset_bias** — signed boundary bias (ms)
- **PER** — phoneme error rate of the transcription front-end (%)

## Status and caveats

This is research code as it ran on the lab server, kept in its original layout for
reproducibility rather than repackaged as a library. Consequences worth knowing:

- Paths to corpora and outputs are **hardcoded absolute paths** (e.g.
  `/vol/corpora/Rhapsodie/wav16k_corrected`) and must be edited for another machine.
- Exploratory scripts and notebooks sit alongside the ones used for the paper; the
  reported results come from `src/utils/w2vctc_rhapsodie.ipynb`.
- `src/utils/trackeval_v2.py` carries a pre-existing indentation error and will not
  import as-is; it is committed unmodified. Use `align_metrics.py` instead.

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
