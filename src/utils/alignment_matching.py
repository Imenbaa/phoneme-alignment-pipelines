"""Pair reference and hypothesis intervals through a Levenshtein alignment.

Only pairs whose phoneme labels are identical contribute a boundary error, so
insertions, deletions and substitutions are excluded by construction. This is
the matching step that sits between normalisation and the metric aggregation in
:mod:`metrics_alignment`.
"""

from .metrics_alignment import align_sequences


def match_alignments_lev(
    ref_alignments,
    hyp_alignments,
    ref_seq,
    hyp_seq,
    max_time_diff=None,
    align_fn=align_sequences,
):
    """Match reference and hypothesis intervals and measure boundary errors.

    Args:
        ref_alignments, hyp_alignments: interval lists, in seconds.
        ref_seq, hyp_seq: the matching phoneme sequences (same length / order).
        max_time_diff: drop a matched pair whose midpoints differ by more than
            this many seconds. ``None`` (default) keeps every matched pair --
            filtering discards the worst errors and inflates the result, so use
            it only for diagnostics, never for reported numbers.
        align_fn: sequence aligner, for swapping in a banded variant.

    Returns:
        (start_errors, end_errors, duration_errors, mid_errors, matched_pairs,
         filtered_count) -- errors are absolute, in SECONDS, one entry per
        matched pair; matched_pairs holds ``(ref_idx, hyp_idx)``.
    """
    alignment = align_fn(ref_seq, hyp_seq)

    start_errors = []
    end_errors = []
    duration_errors = []
    mid_errors = []
    matched_pairs = []
    filtered_count = 0

    for ref_idx, hyp_idx in alignment:
        if ref_idx is None or hyp_idx is None:
            continue
        if ref_seq[ref_idx] != hyp_seq[hyp_idx]:
            continue

        r_start = ref_alignments[ref_idx]["start"]
        r_end = ref_alignments[ref_idx]["end"]
        h_start = hyp_alignments[hyp_idx]["start"]
        h_end = hyp_alignments[hyp_idx]["end"]

        mid_error = abs((r_start + r_end) / 2 - (h_start + h_end) / 2)
        if max_time_diff is not None and mid_error > max_time_diff:
            filtered_count += 1
            continue

        start_errors.append(abs(r_start - h_start))
        end_errors.append(abs(r_end - h_end))
        duration_errors.append(abs((r_end - r_start) - (h_end - h_start)))
        mid_errors.append(mid_error)
        matched_pairs.append((ref_idx, hyp_idx))

    return (
        start_errors,
        end_errors,
        duration_errors,
        mid_errors,
        matched_pairs,
        filtered_count,
    )
