# src/Llama_index_sandbox/utils/store_response.py

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

# ✅ Core LlamaIndex types (no legacy)
from llama_index.core.schema import NodeWithScore  # type: ignore
try:
    # Most current locations
    from llama_index.core.base.response.schema import Response  # type: ignore
except Exception:
    # Fallback if package layout differs
    from llama_index.core.response import Response  # type: ignore


# --------- helpers: safe extraction & JSON-coercion ---------

def _safe_node_id(node: Any) -> str | None:
    # Core nodes typically expose .node_id, some builds keep .id_
    return getattr(node, "node_id", getattr(node, "id_", None))

def _safe_text(node: Any) -> str | None:
    # Prefer .text when available, fallback to get_content() if present
    txt = getattr(node, "text", None)
    if txt is not None:
        return txt
    getter = getattr(node, "get_content", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None

def _safe_metadata(md: Any) -> Dict[str, Any]:
    # Ensure a dict, then coerce values that aren't JSON serializable into strings
    if not isinstance(md, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in md.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out

def _safe_score(nws: Any) -> float | None:
    # Core NodeWithScore: .score; old shims: .get_score()
    if hasattr(nws, "score"):
        return getattr(nws, "score")
    getter = getattr(nws, "get_score", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


# --------- public serializers ---------

def node_with_score_to_dict(node_with_score: NodeWithScore) -> Dict[str, Any]:
    node = getattr(node_with_score, "node", None)
    return {
        "node_id": _safe_node_id(node),
        "score": _safe_score(node_with_score),
        "text": _safe_text(node),
        "metadata": _safe_metadata(getattr(node, "metadata", {})),
    }


def response_to_dict(response: Response) -> Dict[str, Any]:
    source_nodes: List[NodeWithScore] = getattr(response, "source_nodes", []) or []
    metadata = _safe_metadata(getattr(response, "metadata", {}))
    return {
        "response": getattr(response, "response", None),
        "source_nodes": [node_with_score_to_dict(n) for n in source_nodes],
        "metadata": metadata,
    }


# --------- storage ---------

def store_response(
    embedding_model_name: str,
    llm_model_name: str,
    text_splitter_chunk_size: int,
    text_splitter_chunk_overlap_percentage: int,
    query_str: str,
    response: Response,
    *,
    dir_path: str = "datasets/evaluation_results",
    subjective_score: str | None = None,
) -> None:
    """
    Append a record of (query, response, provenance) into a daily JSON file.

    - Uses only core LlamaIndex types
    - Safely serializes metadata and node contents
    - Appends to a per-day file keyed by embedding/LLM/chunk params
    """
    os.makedirs(dir_path, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_name = (
        f"{date_str}_{embedding_model_name}_{llm_model_name}_"
        f"{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}.json"
    )
    file_path = os.path.join(dir_path, file_name)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subjective_score": subjective_score or "N/A",
        "params": {
            "embedding_model_name": embedding_model_name,
            "llm_model_name": llm_model_name,
            "chunk_size": text_splitter_chunk_size,
            "chunk_overlap_pct": text_splitter_chunk_overlap_percentage,
        },
        "query_str": query_str,
        "response": response_to_dict(response),
    }

    # Append to JSON list on disk
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing.append(record)
            else:
                existing = [existing, record]
        except Exception:
            # If file is corrupted or unreadable, start fresh but keep a backup
            backup = file_path + ".bak"
            try:
                os.replace(file_path, backup)
            except Exception:
                pass
            existing = [record]
    else:
        existing = [record]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)
