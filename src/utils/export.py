"""Export alignment results to the formats the downstream tooling reads.

An *alignment store* is the dictionary produced by the evaluation loop:

    {file_id: {"ref_intervals": [...], "hyp_intervals": [...],
               "ref_seq": [...], "hyp_seq": [...]}}

Which side to export is a parameter, so the same writer serves reference and
hypothesis.
"""

import csv
import pickle


def load_alignment_store(pkl_path):
    """Read an alignment store back from a pickle."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def dict_to_csv(alignment_store, output_path="phonemes.csv", key="hyp_intervals"):
    """Write ``file_id, phoneme sequence`` -- one row per file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "phonemes"])
        for filename, content in alignment_store.items():
            phonemes = " ".join(iv["phoneme"] for iv in content.get(key, []))
            writer.writerow([filename, phonemes])
    return output_path


def write_etf(alignment_store, output_path, key="hyp_intervals"):
    """Write intervals in ETF form, one line per (segment, candidate phoneme).

    For each file, every phoneme present in its intervals becomes a target
    class, and every segment is emitted against every target with decision
    ``t`` when the segment carries that phoneme and ``f`` otherwise -- the
    per-class detection timeline the scorer expects.

    Line format: ``file channel start duration sc - phoneme - decision``.
    """
    with open(output_path, "w", encoding="utf-8") as out:
        for filename, content in alignment_store.items():
            intervals = sorted(content.get(key, []), key=lambda x: x["start"])
            if not intervals:
                continue

            targets = sorted({seg["phoneme"] for seg in intervals})
            for target in targets:
                for seg in intervals:
                    duration = seg["end"] - seg["start"]
                    if duration <= 0:
                        continue
                    decision = "t" if seg["phoneme"] == target else "f"
                    out.write(
                        f"{filename} 1 "
                        f"{seg['start']:.6f} {duration:.6f} "
                        f"sc - {target} - {decision}\n"
                    )
    return output_path


def pkl_to_etf(pkl_path, output_path, key="hyp_intervals"):
    """Convenience wrapper: load a pickled alignment store and write ETF."""
    return write_etf(load_alignment_store(pkl_path), output_path, key=key)
