"""Operations on interval lists.

An *interval list* is the common currency between every module here:

    [{"phoneme": str, "start": seconds, "end": seconds}, ...]

sorted by start time. Reference and hypothesis use the same shape, so nothing
below distinguishes them beyond the argument name.
"""

from .audio_io import audio_duration


def extract_phoneme_sequence(intervals):
    """The phoneme tokens of an interval list, index-aligned with it."""
    return [item["phoneme"] for item in intervals]


def copy_intervals(intervals):
    """Shallow copy -- use when a caller may mutate the list in place."""
    return list(intervals)


def shift_intervals(intervals, offset):
    """Return a copy with `offset` seconds subtracted from every timestamp."""
    return [
        {
            "phoneme": item["phoneme"],
            "start": item["start"] - offset,
            "end": item["end"] - offset,
        }
        for item in intervals
    ]


def correct_interval_offset(
    intervals,
    audio_path=None,
    duration=None,
    tolerance=0.5,
    force=False,
    verbose=False,
):
    """Re-anchor intervals whose timestamps are relative to a longer recording.

    Annotation exported from a segmented session often keeps session-level
    timestamps, so the intervals run past the end of the audio file they are
    scored against. When the last interval ends more than `tolerance` seconds
    beyond the audio duration, everything is shifted back by the first
    interval's start time; otherwise the list is returned unchanged.

    Pass ``force=True`` to shift unconditionally (no audio needed), or give
    `duration` instead of `audio_path` when the duration is already known.
    """
    if not intervals:
        return []

    offset = intervals[0]["start"]
    if force:
        return shift_intervals(intervals, offset)

    if duration is None:
        if audio_path is None:
            raise ValueError("pass audio_path, duration, or force=True")
        duration = audio_duration(audio_path)

    last_end = intervals[-1]["end"]
    if last_end > duration + tolerance:
        if verbose:
            print(
                f"  offset detected: last_end={last_end:.2f}s, "
                f"audio={duration:.2f}s, shift={offset:.3f}s"
            )
        return shift_intervals(intervals, offset)
    return copy_intervals(intervals)


def total_duration(intervals):
    """Summed duration of the intervals, in seconds (gaps excluded)."""
    return sum(item["end"] - item["start"] for item in intervals)
