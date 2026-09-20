#!/usr/bin/env python3
"""Evaluate phoneme-alignment quality against a reference segmentation.

This is the evaluation entry point behind the paper's tables, with every corpus
detail exposed as a flag: point it at a directory of audio and a directory of
reference TextGrids and it reports boundary error, boundary F1 and PER. No
dataset is referenced anywhere -- naming scheme, tier names, phoneme mappings
and grouping metadata are all arguments.

Two kinds of hypothesis can be scored, and they produce the same report, which
is what makes the pipelines comparable:

  --hypothesis ctc       run a CTC phoneme model and take its forced-aligned
                         boundaries (wav2vec2 / WavLM / Whisper-encoder)
  --hypothesis textgrid  read boundaries from a directory of TextGrids already
                         produced by another aligner (MFA, Praat, ...)

Examples
--------
Score a wav2vec2 CTC model, no label mapping, VAD chunking:

    python src/evaluate_alignment.py \\
        --audio-dir  data/wav \\
        --ref-dir    data/reference_textgrids \\
        --ref-tier   phones \\
        --hypothesis ctc --model-type wav2vec2 --checkpoint models/w2v2-phonemizer \\
        --chunking   vad \\
        --out-dir    results/w2v2

Score an MFA run against the same references, mapping SAMPA labels to IPA and
reporting per speaking style:

    python src/evaluate_alignment.py \\
        --audio-dir  data/wav \\
        --ref-dir    data/reference_textgrids --ref-tier sampa \\
        --ref-stem-suffix -Pro --ref-preset french-sampa \\
        --hypothesis textgrid --hyp-dir mfa_output --hyp-tier phones \\
        --groups-csv data/style.csv --group-key file --group-value style \\
        --out-dir    results/mfa
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

# Works both as `python src/evaluate_alignment.py` and `python -m src.evaluate_alignment`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import corpus, export  # noqa: E402
from utils.intervals import correct_interval_offset, extract_phoneme_sequence  # noqa: E402
from utils.metrics_alignment import compute_metrics  # noqa: E402
from utils.phoneme_normalization import (  # noqa: E402
    PhonemeNormalizer,
    clean_alignment_dict,
    french_ipa_normalizer,
)
from utils.textgrid_io import read_phone_intervals  # noqa: E402

# utils.ctc_alignment pulls in torch, so it is imported lazily: scoring TextGrid
# output from another aligner must not require a deep-learning stack.


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------
#: Abort after this many consecutive failures while nothing has succeeded.
FAIL_FAST_AFTER = 3

NORMALIZER_PRESETS = {
    "none": lambda: PhonemeNormalizer(),
    "french-sampa": lambda: french_ipa_normalizer("reference"),
    "french-asr-codes": lambda: french_ipa_normalizer("asr_codes"),
    "french-broad": lambda: french_ipa_normalizer("hypothesis"),
}


def build_normalizer(preset, mapping_path=None):
    """A preset, optionally with its mapping table replaced by a JSON file."""
    normalizer = NORMALIZER_PRESETS[preset]()
    if mapping_path:
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError(f"{mapping_path}: expected a JSON object label -> phoneme")
        normalizer = PhonemeNormalizer(
            mapping=mapping,
            canonical_map=normalizer.canonical_map,
            silence_labels=normalizer.silence_labels,
            drop_labels=normalizer.drop_labels,
            drop_phonemes=normalizer.drop_phonemes,
            strip_markup=normalizer.strip_markup,
            strip_length_marks=normalizer.strip_length_marks,
            truncate_unmapped_clusters=normalizer.truncate_unmapped_clusters,
            vowels=normalizer.vowels,
        )
    return normalizer


# ---------------------------------------------------------------------------
# CTC models
# ---------------------------------------------------------------------------
def load_ctc_model(model_type, checkpoint, tokenizer_path=None, device="cuda"):
    """Load a CTC phoneme model and its front-end.

    Returns a dict consumed by :func:`align_with_ctc`. `tokenizer_path` defaults
    to the checkpoint; point it at the training root when the tokenizer files
    (``vocab.json`` ...) were saved there rather than in the checkpoint.
    """
    import torch

    tokenizer_path = tokenizer_path or checkpoint
    bundle = {"type": model_type}

    if model_type == "wav2vec2":
        from transformers import AutoModelForCTC, Wav2Vec2Processor

        model = AutoModelForCTC.from_pretrained(checkpoint)
        processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        bundle.update(processor=processor, feature_extractor=None, tokenizer=None)

    elif model_type == "wavlm":
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2PhonemeCTCTokenizer,
            WavLMForCTC,
        )

        model = WavLMForCTC.from_pretrained(checkpoint)
        bundle.update(
            processor=None,
            feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(checkpoint),
            tokenizer=_phoneme_tokenizer(Wav2Vec2PhonemeCTCTokenizer, tokenizer_path),
        )

    elif model_type == "whisper":
        from transformers import Wav2Vec2PhonemeCTCTokenizer, WhisperFeatureExtractor

        from utils.whisper_ctc_model import WhisperEncoderForCTC

        model = WhisperEncoderForCTC.from_pretrained(checkpoint)
        bundle.update(
            processor=None,
            feature_extractor=WhisperFeatureExtractor.from_pretrained(checkpoint),
            tokenizer=_phoneme_tokenizer(Wav2Vec2PhonemeCTCTokenizer, tokenizer_path),
        )

    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown model type {model_type!r}")

    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU", file=sys.stderr)
        device = "cpu"
    bundle["model"] = model.to(device).eval()
    return bundle


def _phoneme_tokenizer(cls, path):
    tokenizer = cls.from_pretrained(path)
    # Checkpoints trained with this repo use these special tokens.
    tokenizer.unk_token = "[UNK]"
    tokenizer.pad_token = "[PAD]"
    return tokenizer


def align_with_ctc(bundle, audio_path, chunker, decoder, end_mode, verbose=False):
    """Run the loaded model over one file; returns ``(phoneme_string, intervals)``."""
    from utils.ctc_alignment import MIN_CHUNK_SAMPLES, ctc_align_audio

    kwargs = dict(
        chunker=chunker,
        decoder=decoder,
        end_mode=end_mode,
        verbose=verbose,
    )
    if bundle["type"] == "wav2vec2":
        kwargs["min_samples"] = MIN_CHUNK_SAMPLES
        return ctc_align_audio(
            bundle["model"], audio_path, processor=bundle["processor"], **kwargs
        )
    if bundle["type"] == "wavlm":
        kwargs["min_samples"] = MIN_CHUNK_SAMPLES
    return ctc_align_audio(
        bundle["model"],
        audio_path,
        feature_extractor=bundle["feature_extractor"],
        tokenizer=bundle["tokenizer"],
        **kwargs,
    )


def build_chunker(kind, seconds):
    if kind == "none":
        return None

    from utils.ctc_alignment import fixed_window_chunker, make_vad_chunker

    if kind == "vad":
        return make_vad_chunker(max_chunk_duration=seconds)
    return fixed_window_chunker(seconds)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def inventory_report(ref_inventory, hyp_inventory):
    """Compare the phoneme inventories actually seen on each side.

    A large asymmetry here almost always means the two sides use different label
    conventions, which silently depresses PER and boundary recall -- so it is
    worth reading before the metrics.
    """
    only_ref = sorted(ref_inventory - hyp_inventory)
    only_hyp = sorted(hyp_inventory - ref_inventory)

    lines = [
        f"Reference inventory: {len(ref_inventory)} symbols",
        f"Hypothesis inventory: {len(hyp_inventory)} symbols",
        f"Shared: {len(ref_inventory & hyp_inventory)} symbols",
        "",
    ]
    if not only_ref and not only_hyp:
        lines.append("Inventories are identical.")
    else:
        lines.append("Inventories differ.")
        if only_ref:
            lines += ["", "Only in REFERENCE:", "  " + " ".join(only_ref)]
        if only_hyp:
            lines += ["", "Only in HYPOTHESIS:", "  " + " ".join(only_hyp)]
    lines += [
        "",
        "REFERENCE:  " + " ".join(sorted(ref_inventory)),
        "HYPOTHESIS: " + " ".join(sorted(hyp_inventory)),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = p.add_argument_group("corpus")
    data.add_argument("--audio-dir", required=True, help="directory of audio files")
    data.add_argument("--ref-dir", required=True, help="directory of reference TextGrids")
    data.add_argument("--audio-stem-suffix", default=None,
                      help="marker to strip from audio stems before pairing")
    data.add_argument("--ref-stem-suffix", default=None,
                      help="marker to strip from reference stems, e.g. -Pro")
    data.add_argument("--hyp-stem-suffix", default=None,
                      help="marker to strip from hypothesis TextGrid stems")
    data.add_argument("--exclude", nargs="*", default=[],
                      help="stems to leave out of the evaluation")
    data.add_argument("--limit", type=int, default=None,
                      help="evaluate only the first N files (smoke runs)")

    groups = p.add_argument_group("grouping (optional)")
    groups.add_argument("--groups-csv", default=None,
                        help="CSV giving a group label per file; metrics are reported "
                             "per group as well as globally")
    groups.add_argument("--group-key", default="file",
                        help="CSV column holding the file stem (default: file)")
    groups.add_argument("--group-value", default="style",
                        help="CSV column holding the label (default: style)")

    ref = p.add_argument_group("reference tier and labels")
    ref.add_argument("--ref-tier", default=None, help="exact tier name to read")
    ref.add_argument("--ref-tier-contains", default=None,
                     help="read the first tier whose name contains this")
    ref.add_argument("--ref-tier-index", type=int, default=None, help="read tier by position")
    ref.add_argument("--ref-backend", choices=("praatio", "textgrid"), default="praatio")
    ref.add_argument("--ref-preset", choices=sorted(NORMALIZER_PRESETS), default="none",
                     help="label normalisation for the reference (default: none)")
    ref.add_argument("--ref-mapping", default=None,
                     help="JSON object label -> phoneme, replaces the preset's table")
    ref.add_argument("--ref-offset", choices=("auto", "force", "never"), default="auto",
                     help="re-anchor reference timestamps that carry a session offset")
    ref.add_argument("--ref-offset-tolerance", type=float, default=0.5,
                     help="seconds past the audio end that trigger 'auto' (default: 0.5)")

    hyp = p.add_argument_group("hypothesis")
    hyp.add_argument("--hypothesis", choices=("ctc", "textgrid"), required=True)
    hyp.add_argument("--hyp-preset", choices=sorted(NORMALIZER_PRESETS), default="none")
    hyp.add_argument("--hyp-mapping", default=None,
                     help="JSON object label -> phoneme, replaces the preset's table")
    hyp.add_argument("--hyp-dir", default=None,
                     help="[--hypothesis textgrid] directory of aligner output")
    hyp.add_argument("--hyp-tier", default=None)
    hyp.add_argument("--hyp-tier-contains", default=None)
    hyp.add_argument("--hyp-tier-index", type=int, default=None)
    hyp.add_argument("--hyp-backend", choices=("praatio", "textgrid"), default="praatio")

    ctc = p.add_argument_group("CTC model [--hypothesis ctc]")
    ctc.add_argument("--model-type", choices=("wav2vec2", "wavlm", "whisper"))
    ctc.add_argument("--checkpoint", help="model checkpoint directory or hub id")
    ctc.add_argument("--tokenizer", default=None,
                     help="tokenizer directory, if separate from the checkpoint")
    ctc.add_argument("--device", default="cuda")
    ctc.add_argument("--decoder", choices=("forced_align", "greedy"), default="forced_align",
                     help="boundary source; both decode the same phoneme string")
    ctc.add_argument("--end-mode", choices=("span", "contiguous"), default="span",
                     help="span leaves inter-phoneme silence unassigned")
    ctc.add_argument("--chunking", choices=("none", "vad", "fixed"), default="none",
                     help="none runs whole files; vad needs whisperx installed")
    ctc.add_argument("--chunk-seconds", type=float, default=30.0)

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", required=True)
    out.add_argument("--tag", default=None, help="suffix for the output filenames")
    out.add_argument("--f1-tolerance", type=float, nargs="+", default=[0.02, 0.05],
                     help="boundary F1 tolerances in seconds (default: 0.02 0.05)")
    out.add_argument("--per-phoneme", action="store_true",
                     help="also write the per-phoneme breakdown")
    out.add_argument("--etf", action="store_true",
                     help="also export the hypothesis intervals in ETF form")
    out.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.hypothesis == "ctc" and not (args.model_type and args.checkpoint):
        build_parser().error("--hypothesis ctc requires --model-type and --checkpoint")
    if args.hypothesis == "textgrid" and not args.hyp_dir:
        build_parser().error("--hypothesis textgrid requires --hyp-dir")

    tag = args.tag or (args.model_type if args.hypothesis == "ctc" else "textgrid")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_normalizer = build_normalizer(args.ref_preset, args.ref_mapping)
    hyp_normalizer = build_normalizer(args.hyp_preset, args.hyp_mapping)

    # ---- pair the files -------------------------------------------------
    samples, skipped = corpus.discover_pairs(
        audio_dir=args.audio_dir,
        ref_dir=args.ref_dir,
        hyp_dir=args.hyp_dir if args.hypothesis == "textgrid" else None,
        audio_stem_suffix=args.audio_stem_suffix,
        ref_stem_suffix=args.ref_stem_suffix,
        hyp_stem_suffix=args.hyp_stem_suffix,
        exclude=args.exclude,
    )
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(
            "No audio/reference pairs found. Check --audio-dir, --ref-dir and the "
            f"--*-stem-suffix options. Skipped: {skipped or 'nothing'}"
        )

    if args.groups_csv:
        group_map = corpus.load_group_map(args.groups_csv, args.group_key, args.group_value)
        unmatched = corpus.assign_groups(samples, group_map)
        if unmatched:
            print(f"[warn] no group label for {len(unmatched)} file(s): "
                  f"{', '.join(unmatched[:5])}{' ...' if len(unmatched) > 5 else ''}",
                  file=sys.stderr)

    print(f"{len(samples)} file(s) to evaluate; {len(skipped)} skipped")
    for stem, reason in list(skipped.items())[:10]:
        print(f"  skip {stem}: {reason}")

    # ---- load the model once -------------------------------------------
    bundle = chunker = None
    if args.hypothesis == "ctc":
        bundle = load_ctc_model(args.model_type, args.checkpoint, args.tokenizer, args.device)
        chunker = build_chunker(args.chunking, args.chunk_seconds)

    # ---- evaluate -------------------------------------------------------
    alignment_store = {}
    predictions = {}
    ref_inventory, hyp_inventory = set(), set()
    failures = {}
    consecutive_failures = 0

    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample.stem}", flush=True)
        try:
            ref_raw = read_phone_intervals(
                sample.ref_path,
                tier=args.ref_tier,
                contains=args.ref_tier_contains,
                index=args.ref_tier_index,
                backend=args.ref_backend,
            )

            if args.hypothesis == "ctc":
                phoneme_string, hyp_raw = align_with_ctc(
                    bundle, str(sample.audio_path), chunker,
                    args.decoder, args.end_mode, args.verbose,
                )
            else:
                hyp_raw = read_phone_intervals(
                    sample.hyp_path,
                    tier=args.hyp_tier,
                    contains=args.hyp_tier_contains,
                    index=args.hyp_tier_index,
                    backend=args.hyp_backend,
                )
                phoneme_string = " ".join(iv["phoneme"] for iv in hyp_raw)

            clean_ref = clean_alignment_dict(ref_raw, ref_normalizer)
            clean_hyp = clean_alignment_dict(hyp_raw, hyp_normalizer)

            if args.ref_offset != "never" and clean_ref:
                clean_ref = correct_interval_offset(
                    clean_ref,
                    audio_path=str(sample.audio_path),
                    tolerance=args.ref_offset_tolerance,
                    force=(args.ref_offset == "force"),
                    verbose=args.verbose,
                )

            if not clean_ref or not clean_hyp:
                # Raised, not `continue`d, so it counts toward the fail-fast guard:
                # everything normalising away means the mapping is wrong.
                raise ValueError(
                    f"empty after normalisation "
                    f"(ref={len(clean_ref)}, hyp={len(clean_hyp)}); "
                    "check --ref-preset / --hyp-preset and the tier selection"
                )

            ref_seq = extract_phoneme_sequence(clean_ref)
            hyp_seq = extract_phoneme_sequence(clean_hyp)
            ref_inventory.update(ref_seq)
            hyp_inventory.update(hyp_seq)

            alignment_store[sample.stem] = {
                "file": sample.audio_path.name,
                "style": sample.group,
                "ref_intervals": clean_ref,
                "hyp_intervals": clean_hyp,
                "ref_seq": ref_seq,
                "hyp_seq": hyp_seq,
            }
            predictions[sample.stem] = phoneme_string

        except Exception as exc:  # keep going; report at the end
            failures[sample.stem] = f"{type(exc).__name__}: {exc}"
            if args.verbose:
                import traceback

                traceback.print_exc()

        # A tier name or a mapping that is wrong is wrong for every file, so stop
        # early rather than grinding through the whole corpus to report it.
        consecutive_failures = 0 if sample.stem in alignment_store else consecutive_failures + 1
        if consecutive_failures >= FAIL_FAST_AFTER and not alignment_store:
            raise SystemExit(
                f"Aborting: the first {consecutive_failures} files all failed, most "
                f"recently\n  {sample.stem}: {failures[sample.stem]}\n"
                "This usually means a wrong --ref-tier / --hyp-tier or an unreadable "
                "annotation directory. Re-run with --verbose for the traceback."
            )

    if not alignment_store:
        raise SystemExit(f"Nothing could be evaluated. Failures: {failures}")

    # ---- write everything ----------------------------------------------
    pkl_path = out_dir / f"alignment_{tag}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(alignment_store, f)

    inventory_path = out_dir / f"inventory_{tag}.txt"
    report = inventory_report(ref_inventory, hyp_inventory)
    inventory_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report + "\n")

    predictions_path = out_dir / f"predictions_{tag}.csv"
    import csv as _csv

    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["filename", "phonemes"])
        for stem, phonemes in predictions.items():
            writer.writerow([stem, phonemes])

    metrics_path = out_dir / f"metrics_{tag}.csv"
    per_phoneme_path = out_dir / f"per_phoneme_{tag}.csv" if args.per_phoneme else None
    compute_metrics(
        alignment_store,
        str(metrics_path),
        per_phoneme_csv=str(per_phoneme_path) if per_phoneme_path else None,
        f1_tolerances=tuple(args.f1_tolerance),
    )

    written = [pkl_path, inventory_path, predictions_path, metrics_path]
    if per_phoneme_path:
        written.append(per_phoneme_path)
    if args.etf:
        etf_path = out_dir / f"hyp_{tag}.etf"
        export.write_etf(alignment_store, str(etf_path))
        written.append(etf_path)

    print(f"Evaluated {len(alignment_store)}/{len(samples)} file(s).")
    if failures:
        print(f"{len(failures)} file(s) failed:")
        for stem, reason in list(failures.items())[:10]:
            print(f"  {stem}: {reason}")
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
