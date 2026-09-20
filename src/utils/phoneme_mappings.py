"""Optional phoneme-inventory presets used by :mod:`phoneme_normalization`.

These are *presets*, not defaults: the normalizer applies no mapping unless one
is handed to it explicitly. Pick the table that matches the transcription
convention of whatever reference / hypothesis you are evaluating, or supply your
own dictionary of the same shape.

Shapes
------
``*_TO_IPA``      : token -> IPA token, applied once, exact match.
``*_CANONICAL``   : IPA -> IPA, applied after the mapping to collapse variants.
``*_PROJECTION``  : narrow IPA -> broad IPA, to project a model's inventory onto
                    the (usually coarser) reference inventory.
Sets               : labels or phonemes to drop.
"""

# ---------------------------------------------------------------------------
# Labels that are not phonemes in any convention.
# ---------------------------------------------------------------------------
SILENCE_LABELS = frozenset({"sil", "sp", "spn", "sl", "silence", "<sil>", "_"})

NON_SPEECH_LABELS = frozenset({
    "", "_", " ", "%", "?", "??", "0", "=", "#", "##", "<p:>", "<unk>",
})

# ---------------------------------------------------------------------------
# French presets.
# ---------------------------------------------------------------------------
#: SAMPA-style single characters used by many French transcription tools.
SAMPA_TO_IPA = {
    "Z": "ʒ", "A": "a", "S": "ʃ", "R": "ʁ", "r": "ʁ", "N": "ŋ", "J": "ɲ",
    "H": "ɥ", "g": "ɡ", "Z=": "ʒ", "E": "ɛ", "O": "ɔ", "2": "ø", "9": "œ",
    "@": "ə", "@t": "ə", "a~": "ɑ̃", "o~": "ɔ̃", "e~": "ɛ̃", "9~": "ɛ̃",
    "m=": "m", "n=": "n",
}

#: Two-letter ASR codes (one code per French phoneme) -> IPA.
ASR_CODE_TO_IPA = {
    "aa": "a", "bb": "b", "kk": "k", "dd": "d", "jj": "dʒ", "ei": "e",
    "ff": "f", "ii": "i", "yy": "j", "ll": "l", "mm": "m", "nn": "n",
    "au": "o", "pp": "p", "ss": "s", "tt": "t", "ch": "ʃ", "ou": "u",
    "vv": "v", "ww": "w", "uu": "y", "zz": "z", "eu": "ø", "oe": "œ",
    "an": "ɑ̃", "oo": "ɔ", "on": "ɔ̃", "ee": "ə", "ai": "ɛ", "in": "ɛ̃",
    "un": "ɛ̃", "gn": "ɲ", "gg": "ɡ", "uy": "ɥ", "rr": "ʁ", "r": "ʁ",
    "SIL": "_",
}

#: Collapse the nasal vowels a reference may spell with a single combining tilde.
NASAL_CANONICAL = {
    "ã": "ɑ̃", "ẽ": "ɛ̃", "ĩ": "ɛ̃", "ỹ": "ɛ̃", "œ̃": "ɛ̃", "ə̃": "ɛ̃", "ø̃": "ɛ̃",
}

#: Narrow symbols a multilingual model may emit -> nearest French category.
NARROW_TO_BROAD_PROJECTION = {
    "ɪ": "i", "ʊ": "u", "ɨ": "i", "ɜ": "ə", "ʌ": "ɔ", "ɒ": "ɔ", "ɑ": "a",
    "ɣ": "ʁ", "ɹ": "ʁ", "ɾ": "ʁ", "ʎ": "l", "β": "b", "θ": "t", "c": "k",
    "mʲ": "m", "ɟ": "ɲ", "ñ": "ɲ", "ṽ": "v",
}

#: Symbols outside the target inventory; drop rather than score them.
OUT_OF_INVENTORY = frozenset({
    "β", "θ", "ɹ", "ɾ", "ɣ", "ʌ", "ʊ", "ɪ", "ɨ", "ɨ̃", "ɜ", "ɒ", "õ", "ũ",
})

#: Vowel characters, used by the optional cluster-truncation rule.
VOWEL_CHARS = frozenset("aeiouyɛøœɔɑɨɪʊʌɒɜə")

#: Hesitation / filler tokens, French spelling.
FILLER_LABELS = frozenset({"euh", "eh", "hum", "mm"})
