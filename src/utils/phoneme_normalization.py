"""Corpus-agnostic phoneme label normalisation.

A :class:`PhonemeNormalizer` is a declarative description of how to turn raw
transcription labels into comparable phoneme tokens. It applies, in order:

  1. Unicode normalisation (NFC by default).
  2. Annotation-markup stripping -- ``[[...]]``, ``[...]``, ``(...)``, ``*...*``,
     anything after a newline, and pause tokens. These are notation conventions
     shared by most Praat/ELAN style annotation, not properties of one corpus.
  3. Whitespace removal and a drop test against the silence / non-speech sets.
  4. An optional exact-match mapping (e.g. SAMPA -> IPA), then an optional
     canonicalisation pass (e.g. nasal-vowel variants).
  5. Length-mark stripping and a final out-of-inventory drop test.

No mapping is applied unless you pass one; see :mod:`phoneme_mappings` for
ready-made tables, and :func:`french_ipa_normalizer` for a worked example.
Everything returning ``None`` means "this label is not a scorable phoneme".
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from .phoneme_mappings import (
    NON_SPEECH_LABELS,
    SILENCE_LABELS,
    VOWEL_CHARS,
)

# Annotation markup shared across Praat / ELAN style transcription.
_DOUBLE_BRACKET = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_BRACKET = re.compile(r"\[.*?\]", re.DOTALL)
_STRAY_BRACKET = re.compile(r"\[\[|\]\]|\[|\]")
_PAREN = re.compile(r"\(.*?\)", re.DOTALL)
_STARRED = re.compile(r"\*.*?\*", re.DOTALL)
_PAUSE_TOKEN = re.compile(r"\b\w*pause\w*\b", re.IGNORECASE)
_AFTER_NEWLINE = re.compile(r"\n.*", re.DOTALL)
_WHITESPACE = re.compile(r"\s+")
_REPEATED_TILDE = re.compile("̃+")


def strip_annotation_markup(label: str) -> str:
    """Remove bracketed comments, parentheticals, pause markers and trailing lines."""
    label = _AFTER_NEWLINE.sub("", label)
    label = _DOUBLE_BRACKET.sub("", label)
    label = _BRACKET.sub("", label)
    label = _STRAY_BRACKET.sub("", label)
    label = _PAREN.sub("", label)
    label = _STARRED.sub("", label)
    label = _PAUSE_TOKEN.sub("", label)
    return _WHITESPACE.sub(" ", label).strip()


@dataclass(frozen=True)
class PhonemeNormalizer:
    """Declarative label -> phoneme normaliser. Call it on a single label."""

    #: Exact-match table applied first (e.g. SAMPA_TO_IPA or ASR_CODE_TO_IPA).
    mapping: Mapping[str, str] = field(default_factory=dict)
    #: Applied after `mapping` to collapse notational variants (e.g. NASAL_CANONICAL).
    canonical_map: Mapping[str, str] = field(default_factory=dict)
    #: Labels dropped before any mapping (case-insensitive).
    silence_labels: frozenset = SILENCE_LABELS
    #: Extra labels dropped before any mapping (case-insensitive).
    drop_labels: frozenset = NON_SPEECH_LABELS
    #: Phonemes dropped after mapping, e.g. symbols outside the target inventory.
    drop_phonemes: frozenset = frozenset()
    #: Strip annotation markup (see :func:`strip_annotation_markup`).
    strip_markup: bool = True
    #: Remove the IPA length mark.
    strip_length_marks: bool = True
    #: Keep only the first character of an unmapped multi-character token whose
    #: first character is a consonant. Off by default: it is a lossy heuristic
    #: for references that glue clusters together.
    truncate_unmapped_clusters: bool = False
    vowels: frozenset = VOWEL_CHARS
    unicode_form: str = "NFC"

    def __call__(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            return None

        ph = unicodedata.normalize(self.unicode_form, str(label))
        if self.strip_markup:
            ph = strip_annotation_markup(ph)
        ph = ph.replace(" ", "")

        if self._is_dropped_label(ph):
            return None
        # A label made only of combining marks cannot stand on its own.
        if ph and all(unicodedata.combining(c) for c in ph):
            return None

        ph = _REPEATED_TILDE.sub("̃", ph)

        if (
            self.truncate_unmapped_clusters
            and len(ph) > 1
            and ph not in self.mapping
            and ph[0] not in self.vowels
        ):
            ph = ph[0]

        ph = self.mapping.get(ph, ph)
        ph = self.canonical_map.get(ph, ph)

        if self.strip_length_marks:
            ph = ph.replace("ː", "")

        if not ph or ph in self.drop_phonemes or self._is_dropped_label(ph):
            return None
        return ph

    def _is_dropped_label(self, ph: str) -> bool:
        if not ph:
            return True
        lowered = ph.lower()
        return (
            ph in self.drop_labels
            or lowered in self.drop_labels
            or lowered in self.silence_labels
        )


#: Pass-through normaliser: markup stripping and silence removal only.
IDENTITY_NORMALIZER = PhonemeNormalizer()


def french_ipa_normalizer(kind: str = "reference") -> PhonemeNormalizer:
    """Example configuration mapping French SAMPA / ASR codes onto broad IPA.

    kind:
        ``"reference"``  SAMPA-style reference labels -> IPA.
        ``"asr_codes"``  two-letter ASR codes -> IPA.
        ``"hypothesis"`` project a model's narrow IPA onto the broad inventory.
    """
    from .phoneme_mappings import (
        ASR_CODE_TO_IPA,
        FILLER_LABELS,
        NARROW_TO_BROAD_PROJECTION,
        NASAL_CANONICAL,
        OUT_OF_INVENTORY,
        SAMPA_TO_IPA,
    )

    tables = {
        "reference": SAMPA_TO_IPA,
        "asr_codes": ASR_CODE_TO_IPA,
        "hypothesis": NARROW_TO_BROAD_PROJECTION,
    }
    if kind not in tables:
        raise ValueError(f"kind must be one of {sorted(tables)}, got {kind!r}")

    return PhonemeNormalizer(
        mapping=tables[kind],
        canonical_map=NASAL_CANONICAL,
        drop_labels=NON_SPEECH_LABELS | FILLER_LABELS,
        drop_phonemes=OUT_OF_INVENTORY,
        truncate_unmapped_clusters=(kind == "reference"),
    )


def normalize_phoneme_sequence(
    sequence, normalizer: PhonemeNormalizer = IDENTITY_NORMALIZER
):
    """Normalise a phoneme sequence, dropping labels that normalise away.

    `sequence` may be a whitespace-separated string or an iterable of tokens;
    the return type matches the input.
    """
    if isinstance(sequence, str):
        tokens = sequence.strip().split()
        return " ".join(t for t in map(normalizer, tokens) if t)
    return [t for t in map(normalizer, sequence) if t]


def clean_alignment_dict(
    alignment_list: Iterable[dict],
    normalizer: PhonemeNormalizer = IDENTITY_NORMALIZER,
):
    """Normalise the phoneme of every interval, dropping the ones that vanish.

    Timestamps are carried through untouched, so the returned list stays
    index-aligned with the sequence returned by ``extract_phoneme_sequence``.
    """
    cleaned = []
    for item in alignment_list:
        phoneme = normalizer(item["phoneme"])
        if not phoneme:
            continue
        cleaned.append(
            {"phoneme": phoneme, "start": item["start"], "end": item["end"]}
        )
    return cleaned
