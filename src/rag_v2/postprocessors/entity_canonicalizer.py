from typing import List
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from .entity_utils import normalize_text_entities


class EntityCanonicalizer(BaseNodePostprocessor):
    """Rewrites retrieved node text to fix common entity errors."""

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle=None
    ) -> List[NodeWithScore]:
        """Apply text normalization to all retrieved nodes."""
        for n in nodes:
            if n.node is None:
                continue
            # Fix entity errors in the text content
            n.node.text = normalize_text_entities(n.node.text)
        return nodes