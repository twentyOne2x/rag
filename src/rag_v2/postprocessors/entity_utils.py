import re
from typing import Iterable, List

# Entity canonicalization map
ENT_CANON_MAP = {
    "solana": "Solana",
    "sol": "SOL",
    "soul": "SOL",
    "$sol": "SOL",
    "$soul": "SOL",
    "Seoul": "SOL",
    "anza labs": "Anza",
    "firedancer": "Firedancer",
    "anza": "Anza",
    "anza labs": "Anza",
    "firedancer": "Firedancer",
    "frankendancer": "Firedancer",
    "alpenglow": "Alpenglow",
    "aster": "Aster",
    "Salana": "Solana",
    "Cupsy": "Cupsey",
    "cupsy":"Cupsey",
    "haster":"aster",
    "astro":"astro",
}
ENT_CANON_MAP = {k.lower(): v for k, v in ENT_CANON_MAP.items()}


def canon_entity(s: str) -> str:
    """Canonicalize a single entity."""
    key = s.strip().lower()
    return ENT_CANON_MAP.get(key, s)


def canon_entities(ents: Iterable[str]) -> List[str]:
    """Canonicalize a list of entities."""
    return sorted({canon_entity(e) for e in ents if e and e.strip()})


# Regex for finding "Soul" tokens that should be "SOL"
RE_SOUL_TOKEN = re.compile(r"\b(Soul|\$SOUL|soul|\$soul)\b")


def normalize_text_entities(text: str) -> str:
    """
    Replace Soul/$SOUL with SOL when Solana context is nearby.
    This prevents false positives (e.g., "music for the soul").
    """
    out = []
    i = 0
    while True:
        m = RE_SOUL_TOKEN.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start, end = m.span()
        # Check if "Solana" appears within ±40 characters
        window = text[max(0, start-40):min(len(text), end+40)]
        if re.search(r"\bSolana\b", window, flags=re.I):
            out.append(text[i:start])
            out.append("SOL")
        else:
            out.append(text[i:end])  # leave as-is
        i = end
    return "".join(out)