#!/usr/bin/env python3
"""
Segment an audio file into chunks delimited by SIL intervals of a TextGrid tier.

- Parses both long ("ooTextFile") and short TextGrid formats, UTF-8 / UTF-16.
- Every SIL interval is a cut point. Use --min-sil to ignore very short pauses
  instead (e.g. --min-sil 0.15 keeps pauses under 150 ms inside the chunk).
- Reports start / end / duration (+ phone content) of every chunk.
- Optionally writes the wav chunks to disk.

Usage
-----
  python segment_on_sil.py file.TextGrid
  python segment_on_sil.py file.TextGrid --tier sampa --pad 0.05
  python segment_on_sil.py file.TextGrid --wav file.wav --outdir chunks/ --csv chunks.csv
"""

import argparse
import csv
import os
import re
import sys
import wave

try:
    import soundfile as sf  # optional, handles non-PCM16 wav / flac
except ImportError:
    sf = None

DEFAULT_SIL_LABELS = {"SIL","<p:>","[pause]"}

# --------------------------------------------------------------------------- #
# TextGrid parsing
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(
    r'"((?:[^"]|"")*)"'                     # quoted string (Praat doubles inner ")
    r"|([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"  # number
    r"|(<[^>]*>)"                            # <exists>
)
_INDEX_LINE_RE = re.compile(r"^(item|intervals|points)\s*\[\s*\d*\s*\]\s*:?\s*$")


def _read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode %s" % path)


def _tokenize(text):
    """Flatten a TextGrid (long or short format) into an ordered token list."""
    toks = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _INDEX_LINE_RE.match(line):
            continue
        # long format: "key = value" -> keep only the value side
        if "=" in line:
            key, _, rhs = line.partition("=")
            if '"' not in key:
                line = rhs.strip()
        for m in _TOKEN_RE.finditer(line):
            if m.group(1) is not None:
                toks.append(m.group(1).replace('""', '"'))
            elif m.group(2) is not None:
                toks.append(float(m.group(2)))
            else:
                toks.append(m.group(3))
    return toks


def parse_textgrid(path):
    """Return {tier_name: [(xmin, xmax, label), ...]} for interval tiers."""
    toks = _tokenize(_read_text(path))
    # drop the file header ("ooTextFile", "TextGrid")
    i = 0
    while i < len(toks) and isinstance(toks[i], str) and not toks[i].startswith("<"):
        i += 1
        if i >= 2:
            break
    i = 2 if len(toks) >= 2 else 0
    i += 2                                   # global xmin, xmax
    if i < len(toks) and isinstance(toks[i], str) and toks[i].startswith("<"):
        i += 1                               # tiers? <exists>
    n_tiers = int(toks[i]); i += 1

    tiers = {}
    for _ in range(n_tiers):
        tier_class = toks[i]; i += 1
        tier_name = toks[i]; i += 1
        i += 2                               # tier xmin, xmax
        n_items = int(toks[i]); i += 1
        if tier_class == "IntervalTier":
            items = []
            for _ in range(n_items):
                xmin, xmax, label = toks[i], toks[i + 1], toks[i + 2]
                i += 3
                items.append((float(xmin), float(xmax), str(label)))
            tiers[tier_name] = items
        else:                                # TextTier / point tier: skip
            i += 2 * n_items
    return tiers


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def find_chunks(intervals, sil_labels, min_sil, pad=0.0, min_chunk=0.0):
    """Group intervals into speech chunks separated by long-enough silences."""
    is_sep = [
        (lab.strip() in sil_labels) and (e - s) >= min_sil
        for s, e, lab in intervals
    ]
    chunks, i, n = [], 0, len(intervals)

    while i < n:
        if is_sep[i]:
            i += 1
            continue
        j = i
        while j < n and not is_sep[j]:
            j += 1

        # trim short silences sitting at the edges of the chunk
        a, b = i, j
        while a < b and intervals[a][2].strip() in sil_labels:
            a += 1
        while b > a and intervals[b - 1][2].strip() in sil_labels:
            b -= 1
        if a == b:
            i = j
            continue

        start, end = intervals[a][0], intervals[b - 1][1]
        room_before = (intervals[a - 1][1] - intervals[a - 1][0]) if a > 0 else 0.0
        room_after = (intervals[b][1] - intervals[b][0]) if b < n else 0.0
        start -= min(pad, room_before)
        end += min(pad, room_after)

        if end - start >= min_chunk:
            chunks.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "n_phones": b - a,
                "phones": [intervals[k][2] for k in range(a, b)],
            })
        i = j

    for k, c in enumerate(chunks, 1):
        c["idx"] = k
    return chunks


# --------------------------------------------------------------------------- #
# Audio cutting
# --------------------------------------------------------------------------- #

def cut_audio(audio_path, chunks, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    if sf is not None:
        data, sr = sf.read(audio_path)
        for c in chunks:
            s = max(0, int(round(c["start"] * sr)))
            e = min(len(data), int(round(c["end"] * sr)))
            out = os.path.join(outdir, "%s_%04d.wav" % (prefix, c["idx"]))
            sf.write(out, data[s:e], sr)
            paths.append(out)
        return paths

    with wave.open(audio_path, "rb") as w:                 # PCM fallback
        sr, nch, sw, nfr = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        for c in chunks:
            s = max(0, int(round(c["start"] * sr)))
            e = min(nfr, int(round(c["end"] * sr)))
            w.setpos(s)
            frames = w.readframes(e - s)
            out = os.path.join(outdir, "%s_%04d.wav" % (prefix, c["idx"]))
            with wave.open(out, "wb") as o:
                o.setnchannels(nch)
                o.setsampwidth(sw)
                o.setframerate(sr)
                o.writeframes(frames)
            paths.append(out)
    return paths


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("textgrid")
    ap.add_argument("--tier", default="sampa", help="tier name (default: sampa)")
    ap.add_argument("--wav", help="audio file to cut (optional)")
    ap.add_argument("--outdir", default="chunks", help="where to write wav chunks")
    ap.add_argument("--csv", help="write the chunk table to this CSV file")
    ap.add_argument("--sil", default=",".join(sorted(DEFAULT_SIL_LABELS - {""})),
                    help="comma-separated silence labels (default: %(default)s)")
    ap.add_argument("--min-sil", type=float, default=0.2,
                    help="ignore silences shorter than this instead of cutting "
                         "on them (default: 0.0 = cut on every SIL)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="silence (s) kept on each side, clipped to what is available")
    ap.add_argument("--min-chunk", type=float, default=0.0,
                    help="discard chunks shorter than this (s)")
    ap.add_argument("--no-phones", action="store_true", help="hide the phone sequence")
    args = ap.parse_args()

    tiers = parse_textgrid(args.textgrid)
    if args.tier not in tiers:
        sys.exit("tier %r not found. Available: %s" % (args.tier, ", ".join(tiers)))

    sil_labels = set(args.sil.split(",")) | {""}
    chunks = find_chunks(tiers[args.tier], sil_labels, args.min_sil,
                         args.pad, args.min_chunk)
    if not chunks:
        sys.exit("no chunk found (everything labelled as silence?)")

    print("%-5s %9s %9s %9s %7s" % ("chunk", "start", "end", "dur", "phones"))
    for c in chunks:
        line = "%-5d %9.3f %9.3f %9.3f %7d" % (
            c["idx"], c["start"], c["end"], c["duration"], c["n_phones"])
        if not args.no_phones:
            line += "  " + " ".join(c["phones"])
        print(line)

    durs = [c["duration"] for c in chunks]
    print("\n%d chunks | total %.3f s | min %.3f | max %.3f | mean %.3f"
          % (len(durs), sum(durs), min(durs), max(durs), sum(durs) / len(durs)))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["chunk", "start", "end", "duration", "n_phones", "phones"])
            for c in chunks:
                w.writerow([c["idx"], "%.3f" % c["start"], "%.3f" % c["end"],
                            "%.3f" % c["duration"], c["n_phones"], " ".join(c["phones"])])
        print("wrote %s" % args.csv)

    if args.wav:
        prefix = os.path.splitext(os.path.basename(args.wav))[0]
        paths = cut_audio(args.wav, chunks, args.outdir, prefix)
        print("wrote %d wav files to %s/" % (len(paths), args.outdir))


if __name__ == "__main__":
    main()