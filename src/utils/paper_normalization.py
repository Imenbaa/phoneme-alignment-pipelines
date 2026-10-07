"""The French label normalisation used for the paper, ported rule for rule.

:class:`~phoneme_normalization.PhonemeNormalizer` is the general, declarative
normaliser. The paper's tables were produced by three hand-written functions
whose rules it cannot all express -- a fixed list of dropped multi-character
labels, in-label ``<p:>`` / ``@`` / ``?`` rewriting, cluster truncation on the
hypothesis side too, and a handful of annotation-specific spelling fixes. They are
kept here verbatim so the paper's numbers can be reproduced exactly:

    kind="reference"   SAMPA-style references
    kind="hypothesis"  a phoneme recognizer's IPA output, projected to broad IPA
    kind="asr_codes"   references in two-letter ASR codes

Each instance is a callable ``label -> phoneme or None``, like
``PhonemeNormalizer``, and its exact-match table can be swapped with
:meth:`with_mapping`.
"""

import re
import unicodedata

from .phoneme_mappings import (
    ASR_CODE_TO_IPA,
    NARROW_TO_BROAD_PROJECTION,
    NASAL_CANONICAL,
    OUT_OF_INVENTORY,
    SAMPA_TO_IPA,
    VOWEL_CHARS,
)

#: Labels the paper drops outright, before any other rule.
DROPPED_LABELS = frozenset(
    {"_", " ", "sil", "SIL", "spn", "%", "?", "0", "=", "fe~", "Ra~", "sjo~", "~e"}
)
ASR_CODE_DROPPED_LABELS = frozenset(
    {"_", "sil", "spn", "%", "?", "??", "0", "#", "=", "euh", "#erreur#"}
)


class PaperFrenchNormalizer:
    """One of the paper's three French normalisers (see the module docstring)."""

    KINDS = ("reference", "hypothesis", "asr_codes")

    def __init__(self, kind, mapping=None):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind = kind
        default = {
            "reference": SAMPA_TO_IPA,
            "hypothesis": NARROW_TO_BROAD_PROJECTION,
            "asr_codes": ASR_CODE_TO_IPA,
        }[kind]
        self.mapping = dict(default if mapping is None else mapping)
        # Cluster truncation always tests against the reference table, on both sides.
        self._truncation_exempt = SAMPA_TO_IPA if kind == "hypothesis" else self.mapping

    def with_mapping(self, mapping):
        """Same rules, with the exact-match table replaced."""
        return PaperFrenchNormalizer(self.kind, mapping)

    def __call__(self, label):
        if label is None:
            return None
        if self.kind == "asr_codes":
            return self._asr_codes(label)
        return self._sampa_or_ipa(label)

    def _sampa_or_ipa(self, ph):
        if ph in DROPPED_LABELS:
            return None
        ph = unicodedata.normalize("NFC", ph)
        if ph and all(unicodedata.combining(c) for c in ph):
            return None

        ph = ph.replace(" ", "")
        ph = re.sub(r"@t", "ə", ph)
        ph = re.sub(r"[\?\@]", lambda m: "" if m.group() == "?" else "ə", ph)
        ph = ph.replace("<p:>", "")
        ph = re.sub("̃+", "̃", ph)

        if len(ph) > 1 and ph not in self._truncation_exempt and ph[0] not in VOWEL_CHARS:
            ph = ph[0]

        ph = self.mapping.get(ph, ph)
        ph = NASAL_CANONICAL.get(ph, ph)
        ph = ph.replace("ː", "")
        if ph in OUT_OF_INVENTORY:
            return None
        return ph or None

    def _asr_codes(self, ph):
        ph = unicodedata.normalize("NFC", ph)
        ph = re.sub(r"\[\[.*?\]\]", "", ph)
        ph = re.sub(r"\[\[|\]\]", "", ph)
        ph = re.sub(r"\(.*?\)", "", ph)
        ph = re.sub(r"\bNONCORR\b", "", ph)
        ph = re.sub(r"\s+", " ", ph).strip()
        ph = re.sub(r"\n.*", "", ph, flags=re.DOTALL)
        ph = ph.replace("yu", "uy")
        ph = ph.replace("nn+yy", "nn")
        ph = ph.replace("ei\t\t", "ei")
        if "NB sur tDeb" in ph:
            ph = "ei"
        ph = ph.replace("#a", "a")
        ph = ph.replace("kk+", "k")
        ph = re.sub(r"\*.*?\*", "", ph)
        ph = re.sub(r"\[\s*pause\s*\]", "", ph)
        ph = re.sub(r"\b\w*pause\w*\b", "", ph)

        if ph in ASR_CODE_DROPPED_LABELS:
            return None
        if ph and all(unicodedata.combining(c) for c in ph):
            return None

        ph = self.mapping.get(ph, ph)
        ph = ph.replace("dʒ", "ʒ")
        return ph or None
