#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One figure: what the VAD chunking step costs, as a share of all reference
phones. Reads only <prefix>_per_file.csv from diagnose_chunking.py.

    python plot_chunking_summary.py --prefix chunkdiag
"""

import argparse, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="chunkdiag")
p.add_argument("--aas", type=float, default=57.5, help="corpus AAS in ms")
args = p.parse_args()

rows = list(csv.DictReader(open(f"{args.prefix}_per_file.csv")))
n_ref = sum(int(r["n_ref"]) for r in rows)
lost = sum(int(r["n_lost"]) for r in rows)
clipped = sum(int(r["n_clipped"]) for r in rows)
n_chunks = sum(int(r["n_chunks"]) for r in rows)
absorbed_s = sum(float(r["lead_sum"]) + float(r["trail_sum"]) for r in rows)

edge = 2 * n_chunks - lost - clipped          # phones at a chunk boundary
clean = n_ref - lost - clipped - edge
med_abs_ms = 1000 * absorbed_s / max(1, 2 * n_chunks)
aas_cost = (edge / n_ref) * med_abs_ms        # rough AAS contribution

parts = [
    ("Untouched by chunking", clean, "#c8c8c8"),
    (f"At a chunk edge\n(pause absorbed, ~{med_abs_ms:.0f} ms each)", edge, "#2ca02c"),
    ("Clipped\n(part of the audio cut)", clipped, "#ff7f0e"),
    ("Lost\n(never reaches the model)", lost, "#d62728"),
]

fig, ax = plt.subplots(figsize=(12, 3.6))
left = 0
for label, n, c in parts:
    ax.barh(0, n, left=left, color=c, edgecolor="white")
    left += n

ax.set_xlim(0, n_ref)
ax.set_ylim(-0.6, 1.5)
ax.set_yticks([])
ax.set_xlabel(f"reference phonemes (total {n_ref:,})")
ax.set_title("What the VAD chunking step costs", fontsize=13, loc="left")

# labels above the bar, with leader lines for the thin slices
left = 0
for i, (label, n, c) in enumerate(parts):
    pct = 100 * n / n_ref
    x = left + n / 2
    y = 0.55 if i == 0 else (0.75 + 0.3 * (i % 2))
    ax.annotate(f"{label}\n{n:,}  ({pct:.2f}%)",
                xy=(x, 0.42), xytext=(x, y),
                ha="center", va="bottom", fontsize=9, color=c if i else "#444",
                arrowprops=None if i == 0 else
                dict(arrowstyle="-", color=c, lw=.8))
    left += n

ax.text(0.995, -0.42,
        f"{100*(lost+clipped)/n_ref:.2f}% of phones are damaged or unreachable  ·  "
        f"edge absorption adds roughly {aas_cost:.0f} ms to a {args.aas:.0f} ms AAS",
        transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=9,
        color="#555")

for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(f"{args.prefix}_summary.png", dpi=150)
print(f"[saved] {args.prefix}_summary.png")
print(f"clean={clean} edge={edge} clipped={clipped} lost={lost} "
      f"total={n_ref} | AAS cost ~{aas_cost:.1f} ms")