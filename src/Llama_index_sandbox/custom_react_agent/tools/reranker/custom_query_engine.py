# src/Llama_index_sandbox/custom_react_agent/tools/reranker/custom_query_engine.py
from __future__ import annotations

import heapq
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import pandas as pd
import tldextract

# ---- LlamaIndex (core) ----
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.prompts.mixin import PromptDictType

# ---- Project bits ----
from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.constants import DOCUMENT_TYPES
from src.Llama_index_sandbox.utils.site_configs import site_configs
from src.Llama_index_sandbox.utils.utils import load_csv_data


def _console_safe(s) -> str:
    try:
        return str(s).encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
        return repr(s)


def _log_info_safe(msg: str) -> None:
    logging.info(_console_safe(msg))


class CustomQueryEngine(BaseQueryEngine):
    """
    Core-only query engine that:
      1) Retrieves candidates via a core retriever
      2) Applies CSV-based enrichments & score adjustments
      3) Reranks and synthesizes a response (using a core synthesizer)
    """

    # ---------- weights & config paths ----------
    weights_file = f"{root_dir}/datasets/evaluation_data/effective_weights.pkl"

    document_weights: Dict[str, Dict[str, float]] = {
        f"{DOCUMENT_TYPES.ARTICLE.value}_weights": {},
        f"{DOCUMENT_TYPES.YOUTUBE_VIDEO.value}_weights": {
            "Tim Roughgarden Lectures": 0.95,
            "default": float(os.environ.get("DEFAULT_YOUTUBE_VIDEO_WEIGHT", "0.90")),
        },
        f"{DOCUMENT_TYPES.RESEARCH_PAPER.value}_weights": {
            "default": float(os.environ.get("DEFAULT_RESEARCH_PAPER_WEIGHT", "1.5")),
        },
        "unspecified_weights": {"default": 0.8},
    }

    authors_list: Dict[str, List[str]] = {}
    authors_weights: Dict[str, float] = {
        "Ethereum.org": 1.15,
        "default": 1.0,
    }

    keywords_to_penalise: List[str] = []
    edge_case_of_content_always_cited: List[str] = []
    edge_case_set = set(edge_case_of_content_always_cited)

    document_weight_mappings: Dict[str, Dict[str, float]] = {
        key: {source: weight for source, weight in weights.items() if source != "default"}
        for key, weights in document_weights.items()
    }
    default_weights: Dict[str, float] = {
        key: weights.get("default", 1.0) for key, weights in document_weights.items()
    }

    author_weight_mapping: Dict[str, float] = {}
    for firm, authors in authors_list.items():
        weight = authors_weights.get(firm, 1.0)
        for author in authors:
            author_weight_mapping[author] = weight
    author_weight_mapping["default"] = authors_weights.get("default", 1.0)

    # -------------------------------------------------------------------

    def __init__(
        self,
        *,
        retriever: VectorIndexRetriever,
        callback_manager=None,
    ) -> None:
        super().__init__(callback_manager=callback_manager)
        self._retriever = retriever

        # Lightweight core engine to reuse synthesizer
        self._core_engine = RetrieverQueryEngine.from_args(retriever=self._retriever)

        # effective weights
        self.effective_weights: Dict[Tuple, float] = self.load_or_compute_weights(
            document_weight_mappings=self.document_weight_mappings,
            weights_file=self.weights_file,
            authors_list=self.authors_list,
            authors_weights=self.authors_weights,
            recompute_weights=False,
        )

        # CSV loads
        merged_csv_path = f"{root_dir}/datasets/evaluation_data/merged_articles.csv"
        updated_csv_path = f"{root_dir}/datasets/evaluation_data/articles_updated.csv"

        if os.path.exists(merged_csv_path):
            self.merged_df = load_csv_data(merged_csv_path)
            logging.info(f"Loaded merged_articles.csv with {len(self.merged_df)} rows")
        else:
            self.merged_df = pd.DataFrame()
            logging.warning(f"merged_articles.csv not found at {merged_csv_path}, using empty DataFrame")

        if os.path.exists(updated_csv_path):
            self.updated_df = load_csv_data(updated_csv_path)
            logging.info(f"Loaded articles_updated.csv with {len(self.updated_df)} rows")
        else:
            self.updated_df = pd.DataFrame()
            logging.warning(f"articles_updated.csv not found at {updated_csv_path}, using empty DataFrame")

        # env-configurable multipliers
        self.discourse_only_penalty = float(os.environ.get("DISCOURSE_ONLY_PENALTY", "0.50"))
        self.forum_name_in_title_penalty = float(os.environ.get("FORUM_NAME_IN_TITLE_PENALTY", "1.5"))
        self.doc_to_remove = float(os.environ.get("DOC_TO_REMOVE", "0.0"))
        self.keyword_to_penalise_multiplier = float(os.environ.get("KEYWORD_TO_PENALISE_MULTIPLIER", "0.4"))

        self.site_domains = {urlparse(cfg["base_url"]).netloc for cfg in site_configs.values()}

    # ---------------- PromptMixin hooks (to satisfy abstract requirements) ----------------

    def _get_prompts(self) -> Dict[str, any]:  # type: ignore[override]
        # No custom prompts; keep empty to satisfy PromptMixin
        return {}

    def _get_prompt_modules(self) -> Dict[str, any]:  # type: ignore[override]
        # No prompt modules used here
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:  # type: ignore[override]
        # Nothing to update; present for completeness
        return

    # ===================== weights loader =====================

    @classmethod
    def load_or_compute_weights(
        cls,
        document_weight_mappings,
        weights_file,
        authors_list,
        authors_weights,
        recompute_weights: bool = False,
    ) -> Dict[Tuple, float]:
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)

        def precompute_effective_weights(
            document_weight_mappings,
            authors_weights,
            authors_list,
        ) -> Dict[Tuple, float]:
            effective: Dict[Tuple, float] = {}

            # (A) Articles: triplets (type, source_domain, author_url)
            art_key = f"{DOCUMENT_TYPES.ARTICLE.value}_weights"
            for group, authors in authors_list.items():
                group_w = authors_weights.get(group, 1.0)
                for author_url in authors:
                    a_ext = tldextract.extract(author_url)
                    author_domain = f"{a_ext.domain}.{a_ext.suffix}"
                    for source, w in document_weight_mappings.get(art_key, {}).items():
                        s_ext = tldextract.extract(source)
                        source_domain = f"{s_ext.domain}.{s_ext.suffix}"
                        if author_domain == source_domain:
                            effective[(DOCUMENT_TYPES.ARTICLE.value, source, author_url)] = w * group_w

            # (B) Research papers & videos: pairs (type_key, source)
            for dtype_key, doc_map in document_weight_mappings.items():
                if dtype_key != art_key:
                    for source, w in doc_map.items():
                        effective[(dtype_key, source)] = w

            return effective

        try:
            if not recompute_weights and os.path.exists(weights_file):
                with open(weights_file, "rb") as f:
                    return pickle.load(f)
            effective_weight = precompute_effective_weights(
                document_weight_mappings=document_weight_mappings,
                authors_weights=authors_weights,
                authors_list=authors_list,
            )
            with open(weights_file, "wb") as f:
                pickle.dump(effective_weight, f)
            return effective_weight
        except Exception as e:
            _log_info_safe(f"Error while loading or computing weights: {e}")
            return {}

    # ===================== enrichments & penalties =====================

    def _penalise_discourse_or_forum_title(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        merged_links = set()
        updated_links = set()

        if not self.merged_df.empty and "Link" in self.merged_df.columns:
            merged_links = set(self.merged_df["Link"].dropna().unique())
        if not self.updated_df.empty and "article" in self.updated_df.columns:
            updated_links = set(self.updated_df["article"].dropna().unique())

        for nws in nodes:
            if nws.node.metadata.get("document_type", "") == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                continue

            link = (nws.node.metadata.get("pdf_link") or "").strip()
            title = (nws.node.metadata.get("title") or "").strip().lower()

            if link in merged_links and link not in updated_links:
                nws.score = (nws.score or 0.0) * self.discourse_only_penalty

            if "ethereum research" in title or "flashbots collective" in title:
                nws.score = (nws.score or 0.0) * self.forum_name_in_title_penalty

        return nodes

    def _populate_missing_pdf_links(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        title_to_link: Dict[str, str] = {}

        if not self.merged_df.empty and {"Title", "Link"} <= set(self.merged_df.columns):
            title_to_link.update(
                self.merged_df[["Title", "Link"]].dropna().set_index("Title")["Link"].to_dict()
            )
        if not self.updated_df.empty and {"article", "title"} <= set(self.updated_df.columns):
            upd = (
                self.updated_df.rename(columns={"article": "Link", "title": "Title"})[["Title", "Link"]]
                .dropna()
                .set_index("Title")["Link"]
                .to_dict()
            )
            title_to_link.update(upd)

        for nws in nodes:
            if nws.node.metadata.get("document_type", "") == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                continue
            if not nws.node.metadata.get("pdf_link"):
                title = (nws.node.metadata.get("title") or "").strip()
                matched = title_to_link.get(title)
                if matched:
                    nws.node.metadata["pdf_link"] = matched

        return nodes

    def _adjust_scores_by_criteria(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        BOOST = float(os.environ.get("BOOST_SCORE_MULTIPLIER", "1.3"))
        DOCS_BOOST = float(os.environ.get("DOCS_BOOST_SCORE_MULTIPLIER", "1.20"))
        PENALTY = float(os.environ.get("PENALTY_SCORE_MULTIPLIER", "0.85"))
        CHANNEL_BOOST = [os.environ.get("CHANNEL_NAMES_TO_BOOST", "ETHDenver")]
        CHANNEL_PENAL = [os.environ.get("CHANNEL_NAMES_TO_PENALISE", "Chainlink")]
        DATE_THRESHOLD = datetime.strptime("2024-02-01", "%Y-%m-%d")

        for nws in nodes:
            ch = (nws.node.metadata.get("channel_name") or "").strip()
            if ch in CHANNEL_PENAL:
                nws.score = (nws.score or 0.0) * PENALTY
            if ch in CHANNEL_BOOST:
                rd_str = (nws.node.metadata.get("release_date") or "").strip()
                rd = None
                try:
                    rd = datetime.strptime(rd_str, "%Y-%m-%d") if rd_str else None
                except ValueError:
                    rd = None
                if rd and rd > DATE_THRESHOLD:
                    nws.score = (nws.score or 0.0) * BOOST

            pdf_link = (nws.node.metadata.get("pdf_link") or "").strip()
            if not pdf_link:
                continue
            domain = urlparse(pdf_link).netloc
            if domain in self.site_domains or "docs" in pdf_link.lower() or "documentation" in pdf_link.lower():
                nws.score = (nws.score or 0.0) * DOCS_BOOST
                _log_info_safe(
                    f"Boosting doc: [{nws.node.metadata.get('title', 'UNSPECIFIED')}] from [{pdf_link}]"
                )

        return nodes

    def _apply_special_adjustments(self, nws: NodeWithScore) -> None:
        name = nws.node.metadata.get("title", "UNSPECIFIED")
        if name in self.edge_case_set:
            _log_info_safe(f"Applying [{self.doc_to_remove}] score to document: [{name}]")
            nws.score = (nws.score or 0.0) * self.doc_to_remove

        for kw in self.keywords_to_penalise:
            if kw.lower() in name.lower():
                _log_info_safe(f"Penalising keyword: [{kw}] in document: [{name}]")
                nws.score = (nws.score or 0.0) * self.keyword_to_penalise_multiplier

    # ===================== reranker =====================

    def _rerank(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        NUM_CHUNKS = int(os.environ.get("NUM_CHUNKS_RETRIEVED", "10"))
        SCORE_TH = float(os.environ.get("SCORE_THRESHOLD", "0.10"))
        MIN_CHUNKS = int(os.environ.get("MIN_CHUNKS_FOR_RESPONSE", "2"))

        for nws in nodes:
            self._apply_special_adjustments(nws)

        logging.info(f"count nodes before threshold [{len(nodes)}]")
        nodes = [n for n in nodes if (n.score or 0.0) >= SCORE_TH]
        logging.info(f"count nodes AFTER threshold [{len(nodes)}]")

        if not nodes:
            return nodes

        nodes = self._penalise_discourse_or_forum_title(nodes)
        nodes = self._populate_missing_pdf_links(nodes)
        nodes = self._adjust_scores_by_criteria(nodes)

        if len(nodes) < MIN_CHUNKS:
            logging.warning(f"Number of nodes below minimum chunks: {len(nodes)}")
            return []

        # document-type/source weighting
        for nws in nodes:
            score = (nws.score or 0.0)
            doc_type = (nws.node.metadata.get("document_type") or "UNSPECIFIED").lower()

            if doc_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                source = (nws.node.metadata.get("channel_name") or "UNSPECIFIED LINK").strip()
                key = (doc_type + "_weights", source)
                eff = self.effective_weights.get(
                    key,
                    self.document_weights[doc_type + "_weights"].get("default", 1.0),
                )
            else:
                link = (nws.node.metadata.get("pdf_link") or "UNSPECIFIED LINK").strip()
                extracted = tldextract.extract(link)
                domain = ".".join([s for s in [extracted.subdomain, extracted.domain, extracted.suffix] if s])
                author_url = (nws.node.metadata.get("authors") or "UNSPECIFIED LINK").strip()
                key_triplet = (doc_type, domain, author_url)
                eff = self.effective_weights.get(
                    key_triplet,
                    self.document_weights[doc_type + "_weights"].get(
                        domain, self.document_weights[doc_type + "_weights"].get("default", 1.0)
                    ),
                )
            nws.score = score * eff

        if os.environ.get("ENVIRONMENT") == "LOCAL":
            self._log_unique(nodes[:NUM_CHUNKS], f"Top {NUM_CHUNKS} nodes before rerank")

        top = heapq.nlargest(NUM_CHUNKS, nodes, key=lambda x: (x.score or 0.0))
        top = [n for n in top if (n.score or 0.0) >= SCORE_TH]

        if os.environ.get("ENVIRONMENT") == "LOCAL":
            self._log_unique(top, f"Re-ranked top {NUM_CHUNKS} nodes")

        return top

    # ===================== logging =====================

    def _log_unique(self, nodes: List[NodeWithScore], context: str) -> None:
        unique: Dict[Tuple[str, str], Dict] = {}
        by_title_scores: Dict[str, List[float]] = {}

        _log_info_safe(f"Logging unique file names and document types in context: {context}")

        for n in nodes:
            title = n.node.metadata.get("title", "UNSPECIFIED FILE")
            by_title_scores.setdefault(title, []).append(n.score or 0.0)

        doc_type_count: Dict[str, int] = {}
        for n in nodes:
            title = n.node.metadata.get("title", "UNSPECIFIED FILE")
            dtype = n.node.metadata.get("document_type", "UNSPECIFIED")
            file_key = (dtype, title)

            if file_key not in unique:
                unique[file_key] = {
                    "chunk_count": len(by_title_scores.get(title, [])),
                    "index": len(unique) + 1,
                    "scores": by_title_scores.get(title, []),
                }
            doc_type_count[dtype] = doc_type_count.get(dtype, 0) + 1

        for (dtype, title), info in unique.items():
            scores_str = ", ".join(f"{s:.2f}" for s in info["scores"])
            _log_info_safe(
                f"Unique file #{info['index']}: Document Type: [{dtype}], "
                f"Filename: [{title}], Chunks Retrieved: [{info['chunk_count']}], Scores: [{scores_str}]"
            )

        dist = ", ".join(f"{k}: {v}" for k, v in doc_type_count.items())
        _log_info_safe(f"Document Type chunk-distribution: {dist}")

    # ===================== BaseQueryEngine interface =====================

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as qevent:
            with self.callback_manager.event(
                CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: query_bundle.query_str}
            ) as revent:

                nodes: List[NodeWithScore] = self._retriever.retrieve(query_bundle)

                logging.info(f"Initial retrieval returned {len(nodes)} nodes")
                if nodes:
                    scores = [n.score or 0.0 for n in nodes]
                    logging.info(f"Initial scores range: {min(scores):.4f} - {max(scores):.4f}")

                nodes = self._rerank(nodes)
                logging.info(f"After reranking: {len(nodes)} nodes remain")

                revent.on_end(payload={EventPayload.NODES: nodes})

            if not nodes:
                response_str = (
                    "We could not find any results related to your query. "
                    "However, we encourage you to ask questions about Internet Capital Markets (ICM) and Solana. "
                    "Feel free to ask another question!"
                )
                response: Response = Response(response_str, source_nodes=[], metadata={})
            else:
                response = self._core_engine._response_synthesizer.synthesize(  # type: ignore
                    query=query_bundle, nodes=nodes
                )

            qevent.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as qevent:
            with self.callback_manager.event(
                CBEventType.RETRIEVE, payload={EventPayload.QUERY_STR: query_bundle.query_str}
            ) as revent:

                nodes: List[NodeWithScore] = await self._retriever.aretrieve(query_bundle)

                logging.info(f"[async] Initial retrieval returned {len(nodes)} nodes")
                if nodes:
                    scores = [n.score or 0.0 for n in nodes]
                    logging.info(f"[async] Initial scores range: {min(scores):.4f} - {max(scores):.4f}")

                nodes = self._rerank(nodes)
                logging.info(f"[async] After reranking: {len(nodes)} nodes remain")

                revent.on_end(payload={EventPayload.NODES: nodes})

            if not nodes:
                response_str = (
                    "We could not find any results related to your query. "
                    "However, we encourage you to ask questions about Internet Capital Markets (ICM) and Solana. "
                    "Feel free to ask another question!"
                )
                response: Response = Response(response_str, source_nodes=[], metadata={})
            else:
                response = await self._core_engine._response_synthesizer.asynthesize(  # type: ignore
                    query=query_bundle, nodes=nodes
                )

            qevent.on_end(payload={EventPayload.RESPONSE: response})

        return response
