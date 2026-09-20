"""Pairing audio with its annotation, without assuming a naming scheme.

The evaluation loop needs, per utterance: an audio file, a reference TextGrid,
optionally a hypothesis TextGrid, and optionally a group label. How those files
are named is a property of the corpus, so it is expressed here as two knobs --
a stem suffix to strip on each side -- rather than baked into the loop.

    reference/Rhap-D2004-Pro.TextGrid  --ref-stem-suffix -Pro  ->  stem "Rhap-D2004"
    audio/Rhap-D2004.wav                                       ->  stem "Rhap-D2004"
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
TEXTGRID_EXTENSIONS = (".TextGrid", ".textgrid")

#: Group label used when a sample has no group metadata.
DEFAULT_GROUP = "ALL"


@dataclass
class Sample:
    """One evaluation unit."""

    stem: str
    audio_path: Path
    ref_path: Path
    hyp_path: Optional[Path] = None
    group: str = DEFAULT_GROUP
    extras: dict = field(default_factory=dict)


def strip_stem_suffix(stem, suffix=None):
    """Drop a trailing marker from a file stem (``Rhap-D2004-Pro`` -> ``Rhap-D2004``)."""
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def index_by_stem(directory, extensions, stem_suffix=None, recursive=True):
    """Map ``stem -> path`` for every matching file under `directory`.

    Later duplicates are ignored, so the first match for a stem wins; hidden
    files and Jupyter checkpoint directories are skipped.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    extensions = tuple(e.lower() for e in extensions)
    paths = directory.rglob("*") if recursive else directory.glob("*")

    index = {}
    for path in sorted(paths):
        if not path.is_file() or path.name.startswith("."):
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.suffix.lower() not in extensions:
            continue
        index.setdefault(strip_stem_suffix(path.stem, stem_suffix), path)
    return index


def discover_pairs(
    audio_dir,
    ref_dir,
    hyp_dir=None,
    audio_extensions=AUDIO_EXTENSIONS,
    ref_extensions=TEXTGRID_EXTENSIONS,
    hyp_extensions=TEXTGRID_EXTENSIONS,
    audio_stem_suffix=None,
    ref_stem_suffix=None,
    hyp_stem_suffix=None,
    exclude=(),
):
    """Pair audio, reference and (optionally) hypothesis files by stem.

    Returns:
        ``(samples, skipped)`` -- `skipped` maps stem -> reason, so a caller can
        report what was dropped instead of silently evaluating on less data.
    """
    audio = index_by_stem(audio_dir, audio_extensions, audio_stem_suffix)
    refs = index_by_stem(ref_dir, ref_extensions, ref_stem_suffix)
    hyps = index_by_stem(hyp_dir, hyp_extensions, hyp_stem_suffix) if hyp_dir else None

    exclude = set(exclude)
    samples, skipped = [], {}

    for stem in sorted(refs):
        if stem in exclude:
            skipped[stem] = "excluded"
            continue
        if stem not in audio:
            skipped[stem] = "no audio file"
            continue
        if hyps is not None and stem not in hyps:
            skipped[stem] = "no hypothesis TextGrid"
            continue
        samples.append(
            Sample(
                stem=stem,
                audio_path=audio[stem],
                ref_path=refs[stem],
                hyp_path=hyps[stem] if hyps is not None else None,
            )
        )
    return samples, skipped


def load_group_map(csv_path, key_column, value_column):
    """Read ``stem -> group`` from a CSV (e.g. speaking style per file)."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for column in (key_column, value_column):
            if column not in (reader.fieldnames or []):
                raise KeyError(
                    f"column {column!r} not in {csv_path}; have {reader.fieldnames}"
                )
        for row in reader:
            key = (row[key_column] or "").strip()
            if key:
                mapping[key] = (row[value_column] or "").strip() or DEFAULT_GROUP
    return mapping


def assign_groups(samples, group_map, default=DEFAULT_GROUP):
    """Attach a group label to each sample; returns the stems left unmatched.

    A sample matches on its stem or on its audio filename, so the CSV may key on
    either.
    """
    unmatched = []
    for sample in samples:
        group = group_map.get(sample.stem, group_map.get(sample.audio_path.name))
        if group is None:
            group = default
            unmatched.append(sample.stem)
        sample.group = group
    return unmatched
