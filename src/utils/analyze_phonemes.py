#!/usr/bin/env python3
"""
Per-phoneme detection analysis for a Wav2Vec/WavLM + CTC recognizer.

Input: a dict mapping each phoneme -> a stats dict with at least the base
counts {'tar', 'non', 'miss', 'ins'} (durations in seconds). Derived metrics
(recall, precision, F1, miss_rate, false_alarm_rate, error_rate) are recomputed
from the base counts so the analysis is internally consistent even if some
phonemes are missing the derived fields.

Produces:
  - a per-phoneme CSV (sorted by F1)
  - a per-group (articulatory) CSV
  - a printed summary (macro / weighted / micro averages)
  - four diagnostic plots (PNG)

Usage:
    # from a JSON or pickle file
    python analyze_phonemes.py results.json --outdir phoneme_analysis
    python analyze_phonemes.py results.pkl  --outdir phoneme_analysis

    # or import and call directly on an in-memory dict
    from analyze_phonemes import analyze
    analyze(my_results_dict, outdir="phoneme_analysis")
"""

import argparse
import json
import os
import pickle
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / offline server
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# French articulatory grouping (IPA, as produced by espeak/phonemizer).
# Stored NFC-normalised. Nasal vowels use ɑ/ɛ/ɔ/œ + combining tilde (U+0303);
# there is no precomposed form for these, so they stay as 2-codepoint strings.
# Anything not listed lands in "other" — that bucket is where espeak
# language-switching artefacts and foreign-word phonemes will surface.
# --------------------------------------------------------------------------- #
PHONETIC_GROUPS = {
    "oral_vowel":      ["i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə"],
    "nasal_vowel":     ["ɛ̃", "ɑ̃", "ɔ̃", "œ̃"],
    "semivowel":       ["j", "w", "ɥ"],
    "plosive":         ["p", "b", "t", "d", "k", "ɡ", "g"],
    "fricative":       ["f", "v", "s", "z", "ʃ", "ʒ", "ʁ", "x", "h"],
    "nasal_consonant": ["m", "n", "ɲ", "ŋ"],
    "liquid":          ["l"],
}

# build a reverse lookup, NFC-normalised
_PHONE2GROUP = {}
for _grp, _phones in PHONETIC_GROUPS.items():
    for _p in _phones:
        _PHONE2GROUP[unicodedata.normalize("NFC", _p)] = _grp


def phonetic_group(phone: str) -> str:
    return _PHONE2GROUP.get(unicodedata.normalize("NFC", phone), "other")


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def _safe_div(num, den):
    return num / den if den else 0.0


def metrics_from_counts(tar, non, miss, ins):
    """Recompute every derived metric from the four base counts."""
    hits = tar - miss                       # target duration correctly detected
    recall = _safe_div(hits, tar)
    precision = _safe_div(hits, hits + ins)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "hits": hits,
        "recall": recall,
        "precision": precision,
        "F1": f1,
        "miss_rate": _safe_div(miss, tar),
        "false_alarm_rate": _safe_div(ins, non),
        "error_rate": _safe_div(miss + ins, tar + non),
    }


def to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for phone, s in results.items():
        tar = float(s.get("tar", 0.0))
        non = float(s.get("non", 0.0))
        miss = float(s.get("miss", 0.0))
        ins = float(s.get("ins", 0.0))
        row = {"phoneme": phone, "tar": tar, "non": non, "miss": miss, "ins": ins}
        row.update(metrics_from_counts(tar, non, miss, ins))
        row["group"] = phonetic_group(phone)
        rows.append(row)
    df = pd.DataFrame(rows)
    # tar = total reference duration of the phoneme -> natural "support"
    df = df.rename(columns={"tar": "support_tar"})
    df["tar"] = df["support_tar"]  # keep an alias for clarity in counts
    return df.sort_values("F1", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #
def averages(df: pd.DataFrame) -> dict:
    w = df["support_tar"].to_numpy()
    wsum = w.sum()

    def weighted(col):
        return _safe_div((df[col].to_numpy() * w).sum(), wsum)

    # micro: pool the counts, then compute once
    tar, non = df["tar"].sum(), df["non"].sum()
    miss, ins = df["miss"].sum(), df["ins"].sum()
    micro = metrics_from_counts(tar, non, miss, ins)

    return {
        "n_phonemes": len(df),
        "macro_precision": df["precision"].mean(),
        "macro_recall": df["recall"].mean(),
        "macro_F1": df["F1"].mean(),
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "weighted_F1": weighted("F1"),
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_F1": micro["F1"],
    }


def group_rollup(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("group")[["tar", "non", "miss", "ins", "support_tar"]]
        .sum()
        .reset_index()
    )
    derived = agg.apply(
        lambda r: metrics_from_counts(r["tar"], r["non"], r["miss"], r["ins"]),
        axis=1,
        result_type="expand",
    )
    out = pd.concat([agg, derived], axis=1)
    out["n_phonemes"] = df.groupby("group")["phoneme"].count().reindex(out["group"]).values
    return out.sort_values("F1", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _annotate(ax, df, xcol, ycol, fontsize=7):
    for _, r in df.iterrows():
        ax.annotate(r["phoneme"], (r[xcol], r[ycol]),
                    fontsize=fontsize, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")


def plot_support_vs_recall(df, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["support_tar"], df["recall"], s=30, alpha=0.7)
    _annotate(ax, df, "support_tar", "recall")
    ax.set_xscale("log")
    ax.set_xlabel("support  (tar = reference duration, s, log scale)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Recall vs support — is poor recall just data sparsity?")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_det(df, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["false_alarm_rate"], df["miss_rate"], s=30, alpha=0.7)
    _annotate(ax, df, "false_alarm_rate", "miss_rate")
    ax.set_xlabel("false alarm rate  (ins / non)")
    ax.set_ylabel("miss rate  (miss / tar)")
    ax.set_title("DET-style: upper-left = under-predicted, lower-right = over-predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_pr(df, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = 20 + 380 * (df["support_tar"] / df["support_tar"].max())
    sc = ax.scatter(df["recall"], df["precision"], s=sizes, alpha=0.55,
                    c=df["support_tar"], cmap="viridis")
    _annotate(ax, df, "recall", "precision")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Precision vs recall (point size & colour = support)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="support (tar, s)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_f1_bars(df, path, worst_n=25):
    sub = df.sort_values("F1").head(worst_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(sub))))
    ax.barh(sub["phoneme"], sub["F1"], alpha=0.8)
    ax.set_xlabel("F1")
    ax.set_title(f"{len(sub)} worst phonemes by F1")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_group_bars(grp, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(grp))
    ax.bar(x - 0.2, grp["recall"], width=0.2, label="recall")
    ax.bar(x + 0.0, grp["precision"], width=0.2, label="precision")
    ax.bar(x + 0.2, grp["F1"], width=0.2, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(grp["group"], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Performance by articulatory group (count-pooled)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def analyze(results: dict, outdir: str = "phoneme_analysis") -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    df = to_dataframe(results)
    grp = group_rollup(df)
    avg = averages(df)

    # --- save tables ---
    cols = ["phoneme", "group", "support_tar", "non", "miss", "ins",
            "recall", "precision", "F1", "miss_rate", "false_alarm_rate", "error_rate"]
    df[cols].to_csv(os.path.join(outdir, "per_phoneme.csv"), index=False)
    grp.to_csv(os.path.join(outdir, "per_group.csv"), index=False)

    # --- plots ---
    plot_support_vs_recall(df, os.path.join(outdir, "recall_vs_support.png"))
    plot_det(df, os.path.join(outdir, "det_miss_vs_falsealarm.png"))
    plot_pr(df, os.path.join(outdir, "precision_vs_recall.png"))
    plot_f1_bars(df, os.path.join(outdir, "worst_phonemes_f1.png"))
    plot_group_bars(grp, os.path.join(outdir, "group_performance.png"))

    # --- console summary ---
    print("=" * 64)
    print(f"{avg['n_phonemes']} phonemes analysed")
    print("-" * 64)
    print(f"{'':12}{'precision':>11}{'recall':>9}{'F1':>9}")
    for kind in ("macro", "weighted", "micro"):
        print(f"{kind:12}{avg[kind+'_precision']:>11.3f}"
              f"{avg[kind+'_recall']:>9.3f}{avg[kind+'_F1']:>9.3f}")
    print("-" * 64)
    print("macro = unweighted mean over phonemes (rare units count equally)")
    print("weighted = support-weighted mean (by tar)")
    print("micro = counts pooled across phonemes, then computed once")
    macro_gap = avg["weighted_F1"] - avg["macro_F1"]
    print(f"weighted - macro F1 gap = {macro_gap:+.3f} "
          f"({'rare phonemes are dragging macro down' if macro_gap > 0.05 else 'fairly uniform across support'})")

    print("\n" + "=" * 64)
    print("Worst 10 phonemes by F1:")
    print(df.head(0).columns.tolist() and
          df.sort_values("F1").head(10)[["phoneme", "group", "support_tar",
          "recall", "precision", "F1"]].to_string(index=False))

    print("\n" + "=" * 64)
    print("By articulatory group:")
    print(grp[["group", "n_phonemes", "support_tar", "recall",
               "precision", "F1"]].to_string(index=False))

    print("\n" + "=" * 64)
    print(f"CSV + PNG written to: {os.path.abspath(outdir)}")

    # NOTE: substitution/confusion pairs (e.g. is ɛ̃ collapsing into ɛ or ɑ̃?)
    # CANNOT be derived from this summary dict — miss/ins are aggregate
    # durations, not "confused with X". For that you need the frame-level
    # alignment (reference vs decoded phoneme sequence), then build a
    # confusion matrix from aligned pairs. Hook that in here if/when you
    # have the alignments.
    return df


def _load(path):
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to JSON or pickle dict {phoneme: stats}")
    ap.add_argument("--outdir", default="phoneme_analysis")
    args = ap.parse_args()
    analyze(_load(args.results), outdir=args.outdir)