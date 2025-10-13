import re
from typing import Iterable, List, Tuple

from ..config import ENT_CANON_MAP


def canon_entity(s: str) -> str:
    """Canonicalize a single entity."""
    key = s.strip().lower()
    return ENT_CANON_MAP.get(key, s)


def canon_entities(ents: Iterable[str]) -> List[str]:
    """Canonicalize a list of entities."""
    return sorted({canon_entity(e) for e in ents if e and e.strip()})


SPECIAL_CONTEXT_KEYS = {"soul", "$soul"}


def _compile_canonical_patterns() -> List[Tuple[re.Pattern, str]]:
    """
    Precompile regex patterns for entity replacement.
    Longer keys go first so multi-word phrases win over subsets.
    """
    patterns: List[Tuple[re.Pattern, str]] = []
    for key in sorted(ENT_CANON_MAP.keys(), key=len, reverse=True):
        if key in SPECIAL_CONTEXT_KEYS:
            continue
        replacement = ENT_CANON_MAP[key]
        escaped = re.escape(key)
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        patterns.append((pattern, replacement))
    return patterns


CANON_PATTERNS = _compile_canonical_patterns()

# Regex for finding "Soul" tokens that should be "SOL" in-context
RE_SOUL_TOKEN = re.compile(r"\b(Soul|\$SOUL|soul|\$soul)\b")


def normalize_text_entities(text: str) -> str:
    """
    Apply canonical replacements to free-form text so downstream consumers
    see normalized entity names.
    """
    if not text:
        return text

    # Context-aware fix for Soul/$SOUL to avoid unrelated replacements.
    out = []
    i = 0
    while True:
        match = RE_SOUL_TOKEN.search(text, i)
        if not match:
            out.append(text[i:])
            break
        start, end = match.span()
        window = text[max(0, start - 40):min(len(text), end + 40)]
        if re.search(r"\bSolana\b", window, flags=re.I):
            out.append(text[i:start])
            out.append("SOL")
        else:
            out.append(text[i:end])
        i = end
    normalized = "".join(out)

    # Apply global canonical replacements.
    for pattern, replacement in CANON_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    return normalized
