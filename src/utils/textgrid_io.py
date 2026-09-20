"""Reading phone tiers out of TextGrid files.

The tier to read is selected by an explicit rule rather than a hard-coded name,
so the same reader works for MFA output ("phones"), manually corrected tiers, or
any other naming convention:

    tier="phones"          exact tier name
    contains="corr"        first tier whose name contains this (case-insensitive)
    index=0                positional
    (nothing given)        first tier matching DEFAULT_TIER_NAMES, else tier 0

Two back-ends are available because the two libraries tolerate different
malformed files: ``praatio`` (default) and ``textgrid``.
"""

from .phoneme_mappings import SILENCE_LABELS

#: Tier names tried, in order, when the caller gives no selection rule.
DEFAULT_TIER_NAMES = ("phones", "phonemes", "phone", "phon", "segments")


def select_tier_name(tier_names, tier=None, contains=None, index=None):
    """Resolve a tier selection rule against the tier names of a TextGrid."""
    tier_names = list(tier_names)
    if not tier_names:
        raise ValueError("TextGrid has no tiers")

    if tier is not None:
        if tier not in tier_names:
            raise KeyError(f"tier {tier!r} not in {tier_names}")
        return tier

    if contains is not None:
        needle = contains.lower()
        for name in tier_names:
            if needle in name.lower():
                return name
        raise KeyError(f"no tier name contains {contains!r}; have {tier_names}")

    if index is not None:
        return tier_names[index]

    lowered = {name.lower(): name for name in tier_names}
    for candidate in DEFAULT_TIER_NAMES:
        if candidate in lowered:
            return lowered[candidate]
    return tier_names[0]


def read_phone_intervals(
    textgrid_path,
    tier=None,
    contains=None,
    index=None,
    remove_silence=True,
    silence_labels=SILENCE_LABELS,
    backend="praatio",
):
    """Read a phone tier as ``[{"phoneme", "start", "end"}, ...]`` in seconds.

    Labels are returned raw (only stripped of surrounding whitespace); run them
    through a :class:`~phoneme_normalization.PhonemeNormalizer` afterwards.
    """
    if backend == "praatio":
        entries, _ = _read_praatio(textgrid_path, tier, contains, index)
    elif backend == "textgrid":
        entries, _ = _read_textgrid_lib(textgrid_path, tier, contains, index)
    else:
        raise ValueError(f"backend must be 'praatio' or 'textgrid', got {backend!r}")

    intervals = []
    for start, end, label in entries:
        label = label.strip()
        if not label:
            continue
        if remove_silence and label.lower() in silence_labels:
            continue
        intervals.append({"phoneme": label, "start": float(start), "end": float(end)})
    return intervals


def extract_phones_from_textgrid(
    textgrid_path,
    tier=None,
    contains=None,
    index=None,
    remove_silence=True,
    silence_labels=SILENCE_LABELS,
    backend="praatio",
):
    """Same as :func:`read_phone_intervals`, in ``(phones, (start, end, label))`` form."""
    intervals = read_phone_intervals(
        textgrid_path,
        tier=tier,
        contains=contains,
        index=index,
        remove_silence=remove_silence,
        silence_labels=silence_labels,
        backend=backend,
    )
    phones = [iv["phoneme"] for iv in intervals]
    tuples = [(iv["start"], iv["end"], iv["phoneme"]) for iv in intervals]
    return phones, tuples


def list_tier_names(textgrid_path, backend="praatio"):
    """Tier names of a TextGrid -- useful when picking a selection rule."""
    if backend == "praatio":
        from praatio import textgrid as praatio_textgrid

        tg = praatio_textgrid.openTextgrid(
            textgrid_path, includeEmptyIntervals=True, duplicateNamesMode="rename"
        )
        return list(tg.tierNames)

    from textgrid import TextGrid

    tg = TextGrid()
    tg.read(textgrid_path)
    return [t.name for t in tg.tiers]


def _read_praatio(textgrid_path, tier, contains, index):
    from praatio import textgrid as praatio_textgrid

    tg = praatio_textgrid.openTextgrid(
        textgrid_path, includeEmptyIntervals=True, duplicateNamesMode="rename"
    )
    name = select_tier_name(tg.tierNames, tier, contains, index)
    return list(tg.getTier(name).entries), name


def _read_textgrid_lib(textgrid_path, tier, contains, index):
    from textgrid import TextGrid

    tg = TextGrid()
    tg.read(textgrid_path)
    names = [t.name for t in tg.tiers]
    name = select_tier_name(names, tier, contains, index)
    selected = tg.tiers[names.index(name)]
    entries = [(iv.minTime, iv.maxTime, iv.mark) for iv in selected.intervals]
    return entries, name
