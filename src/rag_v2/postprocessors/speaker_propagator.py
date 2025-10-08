# src/rag_v2/postprocessors/speaker_propagator.py
from typing import List
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

class SpeakerPropagator(BaseNodePostprocessor):
    # src/rag_v2/postprocessors/speaker_propagator.py
    def _postprocess_nodes(self, nodes, query_bundle=None):
        for n in nodes:
            md = n.node.metadata or {}
            if not md.get("speaker"):
                title = md.get("title") or md.get("parent_title")
                sp = md.get("speaker_primary") or _guess_from_title(title)
                if sp: md["speaker"] = sp
        return nodes

def _guess_from_title(title: str | None) -> str | None:
    # e.g. "Kyle Samani - ... | TG Podcast" → "Kyle Samani"
    if not title: return None
    parts = title.split("|", 1)[0].split("-", 1)
    cand = parts[0].strip() if parts else None
    return cand if cand and " " in cand and len(cand) <= 40 else None
