import pandas as pd
import numpy as np
from jiwer import process_words
import Levenshtein

def align_sequences(ref, hyp):
    alignment = []
    ops = Levenshtein.editops(ref, hyp)

    ref_idx = hyp_idx = 0
    op_idx = 0

    while ref_idx < len(ref) or hyp_idx < len(hyp):

        if op_idx < len(ops):
            op_type, src_pos, dest_pos = ops[op_idx]

            if op_type == "delete" and src_pos == ref_idx:
                alignment.append((ref_idx, None))
                ref_idx += 1
                op_idx += 1
                continue

            elif op_type == "insert" and dest_pos == hyp_idx:
                alignment.append((None, hyp_idx))
                hyp_idx += 1
                op_idx += 1
                continue

            elif op_type == "replace" and \
                 src_pos == ref_idx and \
                 dest_pos == hyp_idx:
                alignment.append((ref_idx, hyp_idx))
                ref_idx += 1
                hyp_idx += 1
                op_idx += 1
                continue

        # equal case
        if ref_idx < len(ref) and hyp_idx < len(hyp):
            alignment.append((ref_idx, hyp_idx))
            ref_idx += 1
            hyp_idx += 1
        elif ref_idx < len(ref):
            alignment.append((ref_idx, None))
            ref_idx += 1
        elif hyp_idx < len(hyp):
            alignment.append((None, hyp_idx))
            hyp_idx += 1

    return alignment
def match_alignments_lev(ref_alignments, hyp_alignments, ref_seq, hyp_seq,
                         ):

    alignment = align_sequences(ref_seq, hyp_seq)

    start_errors = []
    end_errors = []
    duration_errors = []
    mid_errors = []
    matched_pairs = []

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

        

        start_errors.append(abs(r_start - h_start))
        end_errors.append(abs(r_end - h_end))
        duration_errors.append(
            abs((r_end - r_start) - (h_end - h_start))
        )
        mid_errors.append(mid_error)

    return start_errors, end_errors, duration_errors, mid_errors
    
def compute_f1(ref_boundaries, pred_boundaries, tolerance):
            ref_boundaries = sorted(ref_boundaries)
            pred_boundaries = sorted(pred_boundaries)
            matched_ref = set()
            TP = 0
            for p in pred_boundaries:
                for i, r in enumerate(ref_boundaries):
                    if i in matched_ref:
                        continue
                    if abs(p - r) <= tolerance:
                        TP += 1
                        matched_ref.add(i)
                        break
            FP = len(pred_boundaries) - TP
            FN = len(ref_boundaries) - TP
            return TP, FP, FN
def metrics_rhap(alignment_store,csv_path):
    results = []

    for f, data in alignment_store.items():
        ref_intervals = data["ref_intervals"]
        hyp_intervals = data["hyp_intervals"]
        ref_seq       = data["ref_seq"]
        hyp_seq       = data["hyp_seq"]
        style_label   = data["style"]  # ← from stored data
    
        # F1 — using start boundaries (consistent with start BE)
        ref_starts  = [seg["start"] for seg in ref_intervals]
        pred_starts = [seg["start"] for seg in hyp_intervals]
        TP_0,  FP_0,  FN_0  = compute_f1(ref_starts, pred_starts, tolerance=0.00)
        TP_20, FP_20, FN_20 = compute_f1(ref_starts, pred_starts, tolerance=0.02)
        TP_50, FP_50, FN_50 = compute_f1(ref_starts, pred_starts, tolerance=0.05)
    
        # PER
        out = process_words(" ".join(ref_seq), " ".join(hyp_seq))
    
        # Boundary errors
        start_err, end_err, dur_err, mid_err = \
            match_alignments_lev(ref_intervals, hyp_intervals, ref_seq, hyp_seq)
    
        mid_arr   = np.array(mid_err) * 1000
        start_arr = np.array(start_err) * 1000
        end_arr   = np.array(end_err) * 1000
    
        # Per-file F1
        p_0  = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0
        r_0  = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0
        f1_0 = 2 * p_0 * r_0 / (p_0 + r_0) if (p_0 + r_0) > 0 else 0
        p_20  = TP_20 / (TP_20 + FP_20) if (TP_20 + FP_20) > 0 else 0
        r_20  = TP_20 / (TP_20 + FN_20) if (TP_20 + FN_20) > 0 else 0
        f1_20 = 2 * p_20 * r_20 / (p_20 + r_20) if (p_20 + r_20) > 0 else 0
    
        p_50  = TP_50 / (TP_50 + FP_50) if (TP_50 + FP_50) > 0 else 0
        r_50  = TP_50 / (TP_50 + FN_50) if (TP_50 + FN_50) > 0 else 0
        f1_50 = 2 * p_50 * r_50 / (p_50 + r_50) if (p_50 + r_50) > 0 else 0
    
        results.append({
            "file":                    f,
            "style":                   style_label,  # ← added
            "N_ref":                   len(ref_seq),
            "N_hyp":                   len(hyp_seq),
            "Substitutions":           out.substitutions,
            "Deletions":               out.deletions,
            "Insertions":              out.insertions,
            "PER (%)":                 (out.substitutions + out.deletions + out.insertions) / len(ref_seq) * 100 if len(ref_seq) > 0 else float("nan"),
            "Mean_start_error (ms)":   np.mean(start_arr)   if len(start_arr) > 0 else float("nan"),
            "Median_start_error (ms)": np.median(start_arr) if len(start_arr) > 0 else float("nan"),
            "Mean_end_error (ms)":     np.mean(end_arr)     if len(end_arr)   > 0 else float("nan"),
            "Median_end_error (ms)":   np.median(end_arr)   if len(end_arr)   > 0 else float("nan"),
            "Mean_mid_error (ms)":     np.mean(mid_arr)     if len(mid_arr)   > 0 else float("nan"),
            "Median_mid_error (ms)":   np.median(mid_arr)   if len(mid_arr)   > 0 else float("nan"),
            "Max_mid_error (ms)":      np.max(mid_arr)      if len(mid_arr)   > 0 else float("nan"),
            "% within 20ms":           np.mean(mid_arr <= 20) * 100 if len(mid_arr) > 0 else float("nan"),
            "% within 50ms":           np.mean(mid_arr <= 50) * 100 if len(mid_arr) > 0 else float("nan"),
            "F1@20ms (%)":             f1_20 * 100,
            "F1@50ms (%)":             f1_50 * 100,
            "TP_20": TP_20, "FP_20": FP_20, "FN_20": FN_20,
            "TP_50": TP_50, "FP_50": FP_50, "FN_50": FN_50,
            "F1@0ms (%)":  f1_0 * 100,
            "TP_0": TP_0, "FP_0": FP_0, "FN_0": FN_0,
            "TP_20": TP_20, "FP_20": FP_20, "FN_20": FN_20,
            "TP_50": TP_50, "FP_50": FP_50, "FN_50": FN_50,
        })
    
    df = pd.DataFrame(results)

    def global_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * p * r / (p + r) * 100 if (p + r) > 0 else 0
    
    def aggregate_group(group_df, label, style_val):
        return {
            "file":                    label,
            "style":                   style_val,
            "N_ref":                   group_df["N_ref"].sum(),
            "N_hyp":                   group_df["N_hyp"].sum(),
            "Substitutions":           group_df["Substitutions"].sum(),
            "Deletions":               group_df["Deletions"].sum(),
            "Insertions":              group_df["Insertions"].sum(),
            "PER (%)":                 (group_df["Substitutions"].sum() + group_df["Deletions"].sum() + group_df["Insertions"].sum()) / group_df["N_ref"].sum() * 100,
            "Mean_start_error (ms)":   group_df["Mean_start_error (ms)"].mean(),
            "Median_start_error (ms)": group_df["Median_start_error (ms)"].median(),
            "Mean_end_error (ms)":     group_df["Mean_end_error (ms)"].mean(),
            "Median_end_error (ms)":   group_df["Median_end_error (ms)"].median(),
            "Mean_mid_error (ms)":     group_df["Mean_mid_error (ms)"].mean(),
            "Median_mid_error (ms)":   group_df["Median_mid_error (ms)"].median(),
            "Max_mid_error (ms)":      group_df["Max_mid_error (ms)"].max(),
            "% within 20ms":           group_df["% within 20ms"].mean(),
            "% within 50ms":           group_df["% within 50ms"].mean(),
            "F1@20ms (%)":             global_f1(group_df["TP_20"].sum(), group_df["FP_20"].sum(), group_df["FN_20"].sum()),
            "F1@50ms (%)":             global_f1(group_df["TP_50"].sum(), group_df["FP_50"].sum(), group_df["FN_50"].sum()),
            "F1@0ms (%)": global_f1(group_df["TP_0"].sum(), group_df["FP_0"].sum(), group_df["FN_0"].sum()),
"TP_0": group_df["TP_0"].sum(), "FP_0": group_df["FP_0"].sum(), "FN_0": group_df["FN_0"].sum(),
            "TP_20": group_df["TP_20"].sum(), "FP_20": group_df["FP_20"].sum(), "FN_20": group_df["FN_20"].sum(),
            "TP_50": group_df["TP_50"].sum(), "FP_50": group_df["FP_50"].sum(), "FN_50": group_df["FN_50"].sum(),
        }
    
    # Per-style aggregate rows
    style_rows = []
    for s, group_df in df.groupby("style"):
        style_rows.append(aggregate_group(group_df, label=f"STYLE_{s}", style_val=s))
    
    # Global aggregate row
    global_row = aggregate_group(df, label="GLOBAL", style_val="ALL")
    
    df = pd.concat([df, pd.DataFrame(style_rows), pd.DataFrame([global_row])], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    
def metrics(alignment_store,result_csv):
    results = []

    for f, data in alignment_store.items():
        ref_intervals = data["ref_intervals"]
        hyp_intervals = data["hyp_intervals"]
        ref_seq       = data["ref_seq"]
        hyp_seq       = data["hyp_seq"]
    
        # F1 — using start boundaries (consistent with start BE)
        ref_starts  = [seg["start"] for seg in ref_intervals]
        pred_starts = [seg["start"] for seg in hyp_intervals]
        TP_0,  FP_0,  FN_0  = compute_f1(ref_starts, pred_starts, tolerance=0.00)
        TP_20, FP_20, FN_20 = compute_f1(ref_starts, pred_starts, tolerance=0.02)
        TP_50, FP_50, FN_50 = compute_f1(ref_starts, pred_starts, tolerance=0.05)

        # PER
        out = process_words(" ".join(ref_seq), " ".join(hyp_seq))
    
        # Boundary errors
        start_err, end_err, dur_err, mid_err= \
            match_alignments_lev(ref_intervals, hyp_intervals, ref_seq, hyp_seq)
    
        mid_arr   = np.array(mid_err) * 1000  # convert to ms
        start_arr = np.array(start_err) * 1000
        end_arr   = np.array(end_err) * 1000
    
        # Per-file F1
        p_0  = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) > 0 else 0
        r_0  = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) > 0 else 0
        f1_0 = 2 * p_0 * r_0 / (p_0 + r_0) if (p_0 + r_0) > 0 else 0

        p_20 = TP_20 / (TP_20 + FP_20) if (TP_20 + FP_20) > 0 else 0
        r_20 = TP_20 / (TP_20 + FN_20) if (TP_20 + FN_20) > 0 else 0
        f1_20 = 2 * p_20 * r_20 / (p_20 + r_20) if (p_20 + r_20) > 0 else 0
    
        p_50 = TP_50 / (TP_50 + FP_50) if (TP_50 + FP_50) > 0 else 0
        r_50 = TP_50 / (TP_50 + FN_50) if (TP_50 + FN_50) > 0 else 0
        f1_50 = 2 * p_50 * r_50 / (p_50 + r_50) if (p_50 + r_50) > 0 else 0
    
        results.append({
            "file":                    f,
            "N_ref":                   len(ref_seq),
            "N_hyp":                   len(hyp_seq),
            "Substitutions":           out.substitutions,
            "Deletions":               out.deletions,
            "Insertions":              out.insertions,
            "PER (%)":                 (out.substitutions + out.deletions + out.insertions) / len(ref_seq) * 100 if len(ref_seq) > 0 else float("nan"),
            "Mean_start_error (ms)":   np.mean(start_arr)  if len(start_arr)  > 0 else float("nan"),
            "Median_start_error (ms)": np.median(start_arr) if len(start_arr) > 0 else float("nan"),
            "Mean_end_error (ms)":     np.mean(end_arr)    if len(end_arr)    > 0 else float("nan"),
            "Median_end_error (ms)":   np.median(end_arr)  if len(end_arr)    > 0 else float("nan"),
            "Mean_mid_error (ms)":     np.mean(mid_arr)    if len(mid_arr)    > 0 else float("nan"),
            "Median_mid_error (ms)":   np.median(mid_arr)  if len(mid_arr)    > 0 else float("nan"),
            "Max_mid_error (ms)":      np.max(mid_arr)     if len(mid_arr)    > 0 else float("nan"),
            "% within 20ms":           np.mean(mid_arr <= 20) * 100 if len(mid_arr) > 0 else float("nan"),
            "% within 50ms":           np.mean(mid_arr <= 50) * 100 if len(mid_arr) > 0 else float("nan"),
            "F1@20ms (%)":             f1_20 * 100,
            "F1@50ms (%)":             f1_50 * 100,
            "F1@0ms (%)":  f1_0 * 100,
            "TP_0": TP_0, "FP_0": FP_0, "FN_0": FN_0,
            "TP_20": TP_20, "FP_20": FP_20, "FN_20": FN_20,
            "TP_50": TP_50, "FP_50": FP_50, "FN_50": FN_50,
        })
    
    df = pd.DataFrame(results)
    
    # Global aggregate row
    def global_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * p * r / (p + r) * 100 if (p + r) > 0 else 0
    
    global_row = {
        "file":                    "GLOBAL",
        "N_ref":                   df["N_ref"].sum(),
        "N_hyp":                   df["N_hyp"].sum(),
        "Substitutions":           df["Substitutions"].sum(),
        "Deletions":               df["Deletions"].sum(),
        "Insertions":              df["Insertions"].sum(),
        "PER (%)":                 (df["Substitutions"].sum() + df["Deletions"].sum() + df["Insertions"].sum()) / df["N_ref"].sum() * 100,
        "Mean_start_error (ms)":   df["Mean_start_error (ms)"].mean(),
        "Median_start_error (ms)": df["Median_start_error (ms)"].median(),
        "Mean_end_error (ms)":     df["Mean_end_error (ms)"].mean(),
        "Median_end_error (ms)":   df["Median_end_error (ms)"].median(),
        "Mean_mid_error (ms)":     df["Mean_mid_error (ms)"].mean(),
        "Median_mid_error (ms)":   df["Median_mid_error (ms)"].median(),
        "Max_mid_error (ms)":      df["Max_mid_error (ms)"].max(),
        "% within 20ms":           df["% within 20ms"].mean(),
        "% within 50ms":           df["% within 50ms"].mean(),
        "F1@20ms (%)":             global_f1(df["TP_20"].sum(), df["FP_20"].sum(), df["FN_20"].sum()),
        "F1@50ms (%)":             global_f1(df["TP_50"].sum(), df["FP_50"].sum(), df["FN_50"].sum()),
        "F1@0ms (%)": global_f1(df["TP_0"].sum(), df["FP_0"].sum(), df["FN_0"].sum()),
"TP_0": df["TP_0"].sum(), "FP_0": df["FP_0"].sum(), "FN_0": df["FN_0"].sum(),
        "TP_20": df["TP_20"].sum(), "FP_20": df["FP_20"].sum(), "FN_20": df["FN_20"].sum(),
        "TP_50": df["TP_50"].sum(), "FP_50": df["FP_50"].sum(), "FN_50": df["FN_50"].sum(),
    }
    
    df = pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)
    df.to_csv(result_csv)