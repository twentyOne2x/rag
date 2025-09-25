# src/Llama_index_sandbox/utils/text_sanitizer.py
import re

# Broader set of meta/self-referential phrases we never want in user-visible text.
_META_PATTERNS = [
    # generic "based on/according to the context"
    r"\b(based on|according to|from|within|per)\s+(the\s+)?(provided|above|retrieved)?\s*context\b[,:\s]*",

    # "as discussed/shown/mentioned ... in/by (the) (provided/above/retrieved) context"
    r"\b(as|as was|as is|as discussed|as shown|as outlined|as mentioned)\s+(in|by)\s+(the\s+)?(provided|above|retrieved)?\s*context\b[,:\s]*",

    # direct mentions of tool/retrieval plumbing
    r"\b(the|this)\s+(tool|retrieval|query\s*tool|vector\s*(db|database)|rag\s*(system|tool)?)\s+(says|returned|returns|shows|found|finds|indicates)\b",
    r"\bfrom\s+the\s+(retrieval|query\s*tool|vector\s*(db|database)|rag\s*(system|tool)?)\b",

    # "based on our/my retrieval"
    r"\bbased\s+on\s+(our|my)\s+retrieval\b",

    # "as per the context"
    r"\bas\s+per\s+the\s+context\b",

    # "the/this context shows/suggests/indicates..."
    r"\b(the|this)\s+context\s+(shows|suggests|indicates|provides|contains)\b",

    # "according to the transcript(s)/document/material"
    r"\b(according to|per)\s+(the\s+)?(transcript[s]?|document|material|source[s]?)\b",

    # "the/this answer/response is based on the tool/context"
    r"\b(the|this)\s+(answer|response)\s+(is|was)\s+(based on|from)\s+(the\s+)?(tool|context)\b",
]

# Pre-compile for speed
_META_REGEXES = [re.compile(p, re.IGNORECASE) for p in _META_PATTERNS]


def _strip_meta_once(text: str) -> str:
    out = text
    for rx in _META_REGEXES:
        out = rx.sub("", out)
    return out


def _tidy_whitespace_and_punct(text: str) -> str:
    t = text

    # Remove stray spaces before punctuation: "word , word" -> "word, word"
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)

    # Collapse multiple spaces
    t = re.sub(r"[ \t]{2,}", " ", t)

    # Fix leftover leading commas/colons/semicolons at line starts
    t = re.sub(r"(?m)^[\s]*[,;:–-]\s*", "", t)

    # Collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def strip_meta_phrases(text: str) -> str:
    """
    Remove meta/retrieval/tool scaffolding phrases and tidy punctuation/whitespace.
    Safe to run multiple times; idempotent-ish.
    """
    cleaned = _strip_meta_once(text)
    cleaned = _tidy_whitespace_and_punct(cleaned)
    return cleaned
