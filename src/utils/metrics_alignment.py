"""
Phoneme-alignment metrics, scored exactly as in the paper.

The definitions below reproduce the paper's tables (the `analyse_pkl` scorer
behind the boundary-error and F1 heatmaps):

  - Pairing: Levenshtein (unit-cost) alignment of the reference and hypothesis
    phoneme sequences, with a fixed backtrace order (match/substitution, then
    deletion, then insertion). Only pairs with identical labels contribute a
    boundary error. The backtrace order matters: two minimal alignments can pair
    different phones, so a different aligner gives slightly different numbers.
  - PER: (S + D + I) / N_ref from jiwer, pooled over files.
  - Boundary F1: the boundary set is every phone onset plus the final offset,
    deduplicated. Each REFERENCE boundary takes the NEAREST unused hypothesis
    boundary within the tolerance (one-to-one); TP/FP/FN are pooled over files
    before P, R and F1.
  - AAS / MedianBE: mean / median of the pooled |onset| and |offset| errors of
    correctly-recognised phones, after dropping errors above `cap_ms` (150 ms
    by default) -- larger errors are pairing artifacts, not boundary placement.
  - onset_bias / offset_bias: signed mean onset / offset error (+ = hypothesis
    late), uncapped.

All group stats are POOLED over every phoneme in the group (micro-average).

Diagnostics beyond the paper (uncapped, so the tail stays visible): P90,
%>50ms, %>100ms, %within20ms, medians/means per side, duration error.

Reference point: start/end timestamps are assumed to be in SECONDS on input;
all reported errors are in milliseconds.
"""

import numpy as np
import pandas as pd
from jiwer import process_words

#: Boundary errors above this (ms) are dropped from AAS and MedianBE, as in the paper.
DEFAULT_CAP_MS = 150.0


# ----------------------------------------------------------------------
# Sequence alignment (Levenshtein, unit costs)
# ----------------------------------------------------------------------
def align_sequences(ref, hyp):
    """Return list of (ref_idx, hyp_idx); None on either side = del/ins."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    hyp_a = np.array(hyp, dtype=object)
    for i in range(1, n + 1):
        sub = D[i - 1, :-1] + (hyp_a != ref[i - 1]) if m else D[i - 1, :-1]
        row, prev = D[i], D[i - 1]
        row[0] = i
        for j in range(1, m + 1):
            row[j] = min(sub[j - 1], prev[j] + 1, row[j - 1] + 1)

    i, j, alignment = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            alignment.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            alignment.append((i - 1, None)); i -= 1
        else:
            alignment.append((None, j - 1)); j -= 1
    return alignment[::-1]


# ----------------------------------------------------------------------
# Per-phoneme errors (only phoneme-matched pairs contribute boundary error)
# ----------------------------------------------------------------------
def per_phoneme_errors(ref_alignments, hyp_alignments, ref_seq, hyp_seq, alignment=None):
    """One record per correctly-matched phoneme. Times in seconds in -> ms out."""
    if alignment is None:
        alignment = align_sequences(ref_seq, hyp_seq)
    records = []
    for ref_idx, hyp_idx in alignment:
        if ref_idx is None or hyp_idx is None:
            continue
        if ref_seq[ref_idx] != hyp_seq[hyp_idx]:
            continue
        rs, re = ref_alignments[ref_idx]["start"], ref_alignments[ref_idx]["end"]
        hs, he = hyp_alignments[hyp_idx]["start"], hyp_alignments[hyp_idx]["end"]
        records.append({
            "phoneme":      ref_seq[ref_idx],
            # signed (keep direction): + = hypothesis is LATE
            "signed_start": (hs - rs) * 1000.0,
            "signed_end":   (he - re) * 1000.0,
            # absolute magnitudes
            "start_err":    abs(hs - rs) * 1000.0,
            "end_err":      abs(he - re) * 1000.0,
            "dur_err":      abs((he - hs) - (re - rs)) * 1000.0,
            "mid_err":      abs((hs + he) / 2 - (rs + re) / 2) * 1000.0,
        })
    return records


# ----------------------------------------------------------------------
# Boundary-detection F1 (time-only, label-agnostic).
#
# Boundary set: every phone onset together with the final offset, as in the
#   paper. Offsets before a pause are not counted, so a hypothesis that leaves
#   inter-phoneme gaps (span ends) is scored on its onsets, like a contiguous one.
#
# Matching: each REFERENCE boundary takes the NEAREST unused hypothesis
#   boundary within the tolerance, one-to-one.
#   FP = hypothesis boundaries left unmatched, FN = reference boundaries with
#   no hypothesis boundary within tolerance. A boundary that exists but is
#   outside tolerance therefore costs one FP and one FN.
#
# tolerance is in SECONDS.
# ----------------------------------------------------------------------
def boundary_times(intervals):
    b = [iv["start"] for iv in intervals]
    if intervals:
        b.append(intervals[-1]["end"])
    return sorted(set(b))


def compute_f1_counts(ref_bounds, pred_bounds, tolerance):
    ref_bounds  = np.sort(np.asarray(ref_bounds,  dtype=float))
    pred_bounds = np.sort(np.asarray(pred_bounds, dtype=float))
    used = np.zeros(len(pred_bounds), dtype=bool)
    TP = 0
    for rb in ref_bounds:
        if not len(pred_bounds):
            break
        d = np.abs(pred_bounds - rb)
        d[used] = np.inf
        j = int(d.argmin())
        if d[j] <= tolerance:
            used[j] = True
            TP += 1
    FP = len(pred_bounds) - TP
    FN = len(ref_bounds) - TP
    return TP, FP, FN


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) * 100 if (p + r) > 0 else 0.0


# ----------------------------------------------------------------------
# Pooled boundary statistics over a set of per-phoneme records (ms).
# ----------------------------------------------------------------------
def boundary_stats(df, cap_ms=DEFAULT_CAP_MS):
    if len(df) == 0:
        return {}
    s, e, d = df["start_err"], df["end_err"], df["dur_err"]
    both = pd.concat([s, e])  # AAS / tail pool start AND end boundaries
    capped = both[both <= cap_ms] if cap_ms is not None else both
    return {
        "K":                      int(len(df)),
        # --- central (paper definitions: pooled, capped) ---
        "AAS (ms)":               capped.mean(),             # mean |error| over all boundaries
        "MedianBE (ms)":          capped.median(),           # median |error| over all boundaries
        "onset_bias (ms)":        df["signed_start"].mean(), # + = hyp late
        "offset_bias (ms)":       df["signed_end"].mean(),
        "Median_start (ms)":      s.median(),
        "Mean_start (ms)":        s.mean(),
        "Median_end (ms)":        e.median(),
        "Mean_end (ms)":          e.mean(),
        # --- tail (where MFA fails): pooled over start+end so over-extension shows ---
        "P90 (ms)":               both.quantile(0.90),
        "%>50ms":                 (both > 50).mean() * 100,  # gross-error rate, headline
        "%>100ms":                (both > 100).mean() * 100,
        "%within20ms":            (both <= 20).mean() * 100,
        # --- duration (min-duration over-extension) ---
        "Mean_dur (ms)":          d.mean(),
        "Median_dur (ms)":        d.median(),
        "%dur>50ms":              (d > 50).mean() * 100,
    }


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def compute_metrics(alignment_store, csv_path, per_phoneme_csv=None,
                    f1_tolerances=(0.02, 0.05), cap_ms=DEFAULT_CAP_MS):
    """
    alignment_store: {file_id: {ref_intervals, hyp_intervals, ref_seq, hyp_seq, style}}
        *_intervals: list of {"start": sec, "end": sec}
        *_seq:       list of phoneme tokens
    Writes a file/style/global metrics CSV, and optionally a per-phoneme CSV.
    All boundary stats are pooled (micro-averaged) within each group.
    cap_ms: drop boundary errors above this from AAS / MedianBE (None = keep all).
    """
    all_records = []   # long-form: one row per matched phoneme, tagged with file+style
    f1_rows = []       # per-file F1 counts + PER, pooled later

    for f, data in alignment_store.items():
        ref_intervals = data["ref_intervals"]
        hyp_intervals = data["hyp_intervals"]
        ref_seq, hyp_seq = data["ref_seq"], data["hyp_seq"]
        style = data.get("style", "ALL")

        alignment = align_sequences(ref_seq, hyp_seq)
        recs = per_phoneme_errors(ref_intervals, hyp_intervals, ref_seq, hyp_seq, alignment)
        for r in recs:
            r["file"] = f; r["style"] = style
        all_records.extend(recs)

        # F1 over all boundary times (label-agnostic boundary detection)
        ref_bounds  = boundary_times(ref_intervals)
        pred_bounds = boundary_times(hyp_intervals)
        row = {"file": f, "style": style, "N_ref": len(ref_seq), "N_hyp": len(hyp_seq),
               "B_ref": len(ref_bounds), "B_hyp": len(pred_bounds)}
        for tol in f1_tolerances:
            tp, fp, fn = compute_f1_counts(ref_bounds, pred_bounds, tol)
            tag = f"{int(round(tol * 1000))}"
            row[f"TP_{tag}"], row[f"FP_{tag}"], row[f"FN_{tag}"] = tp, fp, fn

        out = process_words(" ".join(ref_seq), " ".join(hyp_seq))
        row["S"], row["D"], row["I"] = out.substitutions, out.deletions, out.insertions
        f1_rows.append(row)

    records_df = pd.DataFrame(all_records)
    f1_df = pd.DataFrame(f1_rows)
    tags = [f"{int(round(t * 1000))}" for t in f1_tolerances]

    def assemble(rec_subset, f1_subset, label, style_val):
        out = {"group": label, "style": style_val}
        out.update(boundary_stats(rec_subset, cap_ms))
        out["N_ref"] = int(f1_subset["N_ref"].sum())
        out["N_hyp"] = int(f1_subset["N_hyp"].sum())
        out["B_ref"] = int(f1_subset["B_ref"].sum())
        out["B_hyp"] = int(f1_subset["B_hyp"].sum())
        S, D, I = f1_subset["S"].sum(), f1_subset["D"].sum(), f1_subset["I"].sum()
        out["PER (%)"] = (S + D + I) / out["N_ref"] * 100 if out["N_ref"] else np.nan
        out["Deletions"], out["Insertions"] = int(D), int(I)
        for tag in tags:
            tp = f1_subset[f"TP_{tag}"].sum()
            fp = f1_subset[f"FP_{tag}"].sum()
            fn = f1_subset[f"FN_{tag}"].sum()
            out[f"P@{tag}ms (%)"]  = tp / (tp + fp) * 100 if (tp + fp) > 0 else np.nan
            out[f"R@{tag}ms (%)"]  = tp / (tp + fn) * 100 if (tp + fn) > 0 else np.nan
            out[f"F1@{tag}ms (%)"] = _f1(tp, fp, fn)
        return out

    rows = []
    # per file
    for f in f1_df["file"]:
        rsub = records_df[records_df["file"] == f] if len(records_df) else records_df
        fsub = f1_df[f1_df["file"] == f]
        rows.append(assemble(rsub, fsub, f, fsub["style"].iloc[0]))
    # per style
    for s, fsub in f1_df.groupby("style"):
        rsub = records_df[records_df["style"] == s] if len(records_df) else records_df
        rows.append(assemble(rsub, fsub, f"STYLE_{s}", s))
    # global
    rows.append(assemble(records_df, f1_df, "GLOBAL", "ALL"))

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # optional per-phoneme table (pooled over all files), for the §4.3 figure
    if per_phoneme_csv is not None and len(records_df):
        ph_rows = []
        for ph, sub in records_df.groupby("phoneme"):
            r = {"phoneme": ph}; r.update(boundary_stats(sub, cap_ms)); ph_rows.append(r)
        pd.DataFrame(ph_rows).sort_values("AAS (ms)", ascending=False)\
            .to_csv(per_phoneme_csv, index=False)

    return df