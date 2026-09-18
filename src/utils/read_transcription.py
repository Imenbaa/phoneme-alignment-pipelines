
from textgrid import TextGrid, IntervalTier,PointTier

import re

PHONE_LIKE_TOKENS = {
    "a", "e", "i", "o", "u", "y",
    "p", "b", "t", "d", "k", "g",
    "s", "z", "f", "v", "m", "n", "l", "r"
}

WORD_TIER_KEYWORDS = [
    "word", "words", "trans", "orth", "sentence", "text"
]

def is_phone_like(text):
    """Heuristic: phones are short, no spaces, often single chars"""
    text = text.strip().lower()
    if not text:
        return True
    if " " in text:
        return False
    if len(text) <= 2:
        return True
    return text in PHONE_LIKE_TOKENS


def tier_score(tier: IntervalTier):
    """
    Higher score = more likely to be a word tier
    """
    texts = [i.mark.strip() for i in tier.intervals if i.mark.strip()]
    if not texts:
        return -1

    avg_len = sum(len(t) for t in texts) / len(texts)
    space_ratio = sum(" " in t for t in texts) / len(texts)
    phone_ratio = sum(is_phone_like(t) for t in texts) / len(texts)

    name_bonus = any(k in tier.name.lower() for k in WORD_TIER_KEYWORDS)

    score = (
        avg_len * 0.5 +
        space_ratio * 10 -
        phone_ratio * 5 +
        (5 if name_bonus else 0)
    )
    return score
def read_transcription(filename):
    with open(filename, 'r') as f:
        text = f.read()

    return text


def read_preprocess_transcription(filename):
    with open(filename,encoding="utf-8", errors="ignore") as f:
        text = f.read()
    match = re.split(r'\*{3}', text, maxsplit=1)
    after_stars = match[1].strip() if len(match) > 1 else ""

    return after_stars

def get_textgrid_transcription(tg_path):
    tg = TextGrid.fromFile(tg_path)

    candidate_tiers = [
        t for t in tg.tiers if isinstance(t, IntervalTier)
    ]

    if not candidate_tiers:
        return ""

    # Rank tiers by likelihood of being word tiers
    scored = [(tier_score(t), t) for t in candidate_tiers]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_tier = scored[0]

    # Safety check: avoid pure phone tiers
    if best_score < 0:
        return ""

    transcription = " ".join(
        i.mark.strip()
        for i in best_tier.intervals
        if i.mark.strip()
    )

    return transcription

def get_textgrid_transcription_rhap(tg_path, tier_name="word"):
    tg = TextGrid.fromFile(tg_path)

    # Afficher les tiers disponibles (debug)
    available_tiers = [t.name for t in tg.tiers]

    # Chercher le bon tier
    tier = None
    for t in tg.tiers:
        if t.name.strip().lower() == tier_name.strip().lower():
            tier = t
            break

    if tier is None:
        raise ValueError(
            f"Tier '{tier_name}' introuvable. "
            f"Tiers disponibles: {available_tiers}"
        )

    if not isinstance(tier, IntervalTier):
        raise ValueError(f"Le tier '{tier.name}' n'est pas un IntervalTier")

    words = [
        interval.mark.strip()
        for interval in tier.intervals
        if interval.mark.strip()
    ]

    return " ".join(words)
def get_textgrid_transcription_rhap_chunk(tg_path, tier_name="word"):
    tg = TextGrid.fromFile(tg_path)

    # Afficher les tiers disponibles (debug)
    available_tiers = [t.name for t in tg.tiers]

    # Chercher le bon tier
    tier = None
    for t in tg.tiers:
        if t.name.strip().lower() == tier_name.strip().lower():
            tier = t
            break

    if tier is None:
        raise ValueError(
            f"Tier '{tier_name}' introuvable. "
            f"Tiers disponibles: {available_tiers}"
        )

    if not isinstance(tier, IntervalTier):
        raise ValueError(f"Le tier '{tier.name}' n'est pas un IntervalTier")

    words = []
    for interval in tier.intervals:
        if interval.mark.strip():
            words.append(
                (interval.mark.strip(),
                 interval.minTime,
                 interval.maxTime)
            )
    return words
def get_textgrid_transcription_chunk(tg_path, tier_name="transcription"):
    tg = TextGrid.fromFile(tg_path)
    tier = None
    for t in tg.tiers:
        if t.name.lower() == tier_name.lower():
            tier = t
            break
    if tier is None:
        tier = tg.tiers[0]
    words = []
    for interval in tier.intervals:
        if interval.mark.strip():
            words.append(
                (interval.mark.strip(),
                 interval.minTime,
                 interval.maxTime)
            )
    return words
def get_textgrid_transcription_tapas(tg_path, tier_name="transcription"):
    """ Read a TextGrid file and return the concatenated transcription from the specified tier. """
    tg = TextGrid.fromFile(tg_path)
    tier = None
    for t in tg.tiers:
        if t.name.lower() == tier_name.lower():
            tier = t
            break
    if tier is None:
        tier = tg.tiers[0]
    words = [ interval.mark.strip() for interval in tier.intervals if interval.mark and interval.mark.strip() ]

    transcription = " ".join(words)
    return transcription



def read_transcription_text_from_textgrid(tg_path):
    """
    Robustly extract transcription text from a TextGrid.
    Uses CONTENT heuristics instead of tier names.
    Returns: str
    """
    tg = TextGrid()
    tg.read(tg_path)

    def looks_like_words(labels):
        """
        Heuristic:
        - words are longer than 2 chars OR
        - contain apostrophes / accents
        - phones are short (zz, tt, ii, pp, eu...)
        """
        if not labels:
            return False

        phone_like = 0
        for l in labels:
            if re.fullmatch(r"[a-z]{1,2}", l):
                phone_like += 1

        return phone_like / len(labels) < 0.4  # majority are NOT phones

    best_labels = None

    for tier in tg.tiers:
        # Skip acoustic/state tiers
        if ".hmm" in tier.name.lower():
            continue

        labels = []

        if isinstance(tier, IntervalTier):
            labels = [i.mark.strip() for i in tier.intervals if i.mark.strip()]

        elif isinstance(tier, PointTier):
            labels = [p.mark.strip() for p in tier.points if p.mark.strip()]

        if looks_like_words(labels):
            best_labels = labels
            break

    if best_labels is None:
        raise ValueError(
            f"Could not find a word-level transcription. "
            f"Available tiers: {tg.getNames()}"
        )

    return " ".join(best_labels)

import re

import re

def clean_text(text):
    # enlever IDs type gpd_555 / gpf_568
    text = re.sub(r"\b[gG]p[df]_\d+\b", "", text)

    # garder le mot dans [mot, tag]
    text = re.sub(r"\[([^\],]+),[^\]]+\]", r"\1", text)

    # supprimer annotations (R), (@), etc.
    text = re.sub(r"\([^)]*\)", "", text)

    # supprimer rires
    text = re.sub(r"\*+[^*]+\*+", "", text)

    # ❗ supprimer contenu phonétique entre $
    text = re.sub(r"\$[^$]+\$", "", text)

    # ❗ supprimer tokens phonétiques isolés (aa kk jj etc.)
    text = re.sub(r"\b[a-z]{1,2}\b", "", text)

    # ❗ supprimer #a
    text = re.sub(r"#\w+", "", text)

    # normaliser espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_words_text(textgrid_path):
    words = []

    with open(textgrid_path, "r", encoding="utf-16") as f:
        for line in f:
            line = line.strip()

            if not line.startswith("text ="):
                continue

            text = line.split("=", 1)[1].strip().strip('"')

            # ignorer silences
            if text == "#" or not text:
                continue

            text = clean_text(text)

            if text:
                words.extend(text.split())

    return " ".join(words)


