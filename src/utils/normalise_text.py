import re
import unicodedata


def normalization(text):
    # 1. Unicode normalize
    text = unicodedata.normalize("NFKD", text)

    # 2. Remove accents (diacritics)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # 3. Remove text inside square brackets
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # 4. Remove text inside parentheses
    text = re.sub(r"\([^)]*\)", " ", text)

    # 5. Remove symbols & punctuation
    text = "".join(
        " " if unicodedata.category(ch)[0] in {"S", "P"} else ch
        for ch in text
    )

    # 6. Lowercase
    text = text.lower()

    # 7. Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text.split(" ")
def remove_words(trans):

    #respiration
    #externe
    #exterieur
    words = trans.split()

    clean_words = [w for w in words if not ("*" in w and w.count("*") >= 2)]

    res = " ".join(clean_words)
    return res

def clean_french_disfluencies(text: str) -> str:
    # 1. Normalisation unicode
    text = unicodedata.normalize("NFKC", text)

    # 2. Passage en minuscules (optionnel mais conseillé)
    text = text.lower()

    # 3. Liste des hésitations / tics de langage fréquents
    disfluencies = [

        r"\beuh+\b",
        r"\bbah+\b",
        r"\beh+\b",
        r"\bhein+\b",
        r"\bhm+\b",
        r"\bhum+\b",
        r"\bben+\b",
        r"\bmh+\b",
        r"\bheu+\b",
    ]

    # 4. Suppression des hésitations
    for pattern in disfluencies:
        text = re.sub(pattern, " ", text)

    # 6. Nettoyage des espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_french_disfluencies_repetition(text: str) -> str:
    # 1. Normalisation unicode
    text = unicodedata.normalize("NFKC", text)

    # 2. Passage en minuscules (optionnel mais conseillé)
    text = text.lower()

    # 3. Liste des hésitations / tics de langage fréquents
    disfluencies = [
        r"\beuh+\b",
        r"\bbah+\b",
        r"\beh+\b",
        r"\bhein+\b",
        r"\bhm+\b",
        r"\bhum+\b",
        r"\bben+\b",
        r"\bmh+\b",
    ]

    # 4. Suppression des hésitations
    for pattern in disfluencies:
        text = re.sub(pattern, " ", text)

    # 5. Suppression des répétitions de mots (ex: "je je", "c'est c'est")
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text)

    # 6. Nettoyage des espaces
    text = re.sub(r"\s+", " ", text).strip()

    return text