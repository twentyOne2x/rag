import logging
import pickle
from datetime import datetime
from itertools import product

import pandas as pd
import tldextract
from llama_index.legacy import QueryBundle
from llama_index.legacy.callbacks import EventPayload, CBEventType
from llama_index.legacy.indices.base_retriever import BaseRetriever
from llama_index.legacy.response.schema import (
    Response,
)
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.response.schema import RESPONSE_TYPE
from llama_index.legacy.schema import NodeWithScore
from typing import List
import os
import heapq
from urllib.parse import urlparse
from tldextract import extract  # You might need to install this package

from src.Llama_index_sandbox.utils.site_configs import site_configs
from src.Llama_index_sandbox.constants import DOCUMENT_TYPES
from src.Llama_index_sandbox.utils.utils import root_directory, load_csv_data


class CustomQueryEngine(RetrieverQueryEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.effective_weights = self.load_or_compute_weights(
            document_weight_mappings=self.document_weight_mappings,
            weights_file=self.weights_file,
            authors_list=self.authors_list,
            authors_weights=self.authors_weights
        )

        # Safe loading of CSV files
        merged_csv_path = f"{root_directory()}/datasets/evaluation_data/merged_articles.csv"
        updated_csv_path = f"{root_directory()}/datasets/evaluation_data/articles_updated.csv"

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

        self.discourse_only_penalty = float(os.environ.get('DISCOURSE_ONLY_PENALTY', '0.50'))
        self.forum_name_in_title_penalty = float(os.environ.get('FORUM_NAME_IN_TITLE_PENALTY', '1.5'))
        self.doc_to_remove = float(os.environ.get('DOC_TO_REMOVE', '0.0'))
        self.keyword_to_penalise_multiplier = float(os.environ.get('KEYWORD_TO_PENALISE_MULTIPLIER', '0.4'))
        self.site_domains = {urlparse(config['base_url']).netloc for config in site_configs.values()}

    weights_file = f"{root_directory()}/datasets/evaluation_data/effective_weights.pkl"
    document_weights = {
        f'{DOCUMENT_TYPES.ARTICLE.value}_weights': {
        },
        f'{DOCUMENT_TYPES.YOUTUBE_VIDEO.value}_weights': {
            'Tim Roughgarden Lectures': 0.95,
            'default': float(os.environ.get('DEFAULT_YOUTUBE_VIDEO_WEIGHT', '0.90'))
        },
        f'{DOCUMENT_TYPES.RESEARCH_PAPER.value}_weights': {
            'default': float(os.environ.get('DEFAULT_RESEARCH_PAPER_WEIGHT', '1.5'))
        },
        f'unspecified_weights': {  # default case for absent metadata
            'default': 0.8
        }
    }
    authors_list = {
    }
    authors_weights = {
        'Ethereum.org': 1.15,
        'default': 1,
    }

    keywords_to_penalise = []
    edge_case_of_content_always_cited = []

    edge_case_set = set(edge_case_of_content_always_cited)

    # Pre-compute mappings for document weights
    document_weight_mappings = {
        key: {source: weight for source, weight in weights.items() if source != 'default'}
        for key, weights in document_weights.items()
    }
    default_weights = {
        key: weights.get('default', 1) for key, weights in document_weights.items()
    }

    # Pre-compute an author-to-weight mapping
    author_weight_mapping = {}
    for firm, authors in authors_list.items():
        weight = authors_weights.get(firm, 1)
        for author in authors:
            author_weight_mapping[author] = weight

    # Add a default weight
    author_weight_mapping['default'] = authors_weights.get('default', 1)

    # File to store the precomputed effective weights
    weights_file = f"{root_directory()}/datasets/evaluation_data/effective_weights.pkl"

    @classmethod
    def load_or_compute_weights(cls, document_weight_mappings, weights_file, authors_list, authors_weights, recompute_weights=False):
        # Use cls.weights_file instead of CustomQueryEngine.weights_file
        os.makedirs(os.path.dirname(cls.weights_file), exist_ok=True)

        def precompute_effective_weights(document_weight_mappings, authors_weights, authors_list):
            effective_weights = {}

            # Generate all possible combinations for articles with triplets
            for group, authors in authors_list.items():
                group_weight = authors_weights.get(group, 1)
                for author_url in authors:
                    author_extracted = tldextract.extract(author_url)
                    author_domain = author_extracted.domain + '.' + author_extracted.suffix

                    for source, weight in document_weight_mappings.get(DOCUMENT_TYPES.ARTICLE.value + '_weights', {}).items():
                        source_extracted = tldextract.extract(source)
                        source_domain = source_extracted.domain + '.' + source_extracted.suffix

                        # Match domains for articles and calculate weight
                        if author_domain == source_domain:
                            key = (DOCUMENT_TYPES.ARTICLE.value, source, author_url)
                            effective_weights[key] = weight * group_weight

            # Generate pairs for research papers and videos
            for document_type, document_weights in document_weight_mappings.items():
                if document_type != DOCUMENT_TYPES.ARTICLE.value + '_weights':
                    for source, weight in document_weights.items():
                        key = (document_type, source)
                        effective_weights[key] = document_weights.get(source, document_weights.get('default', 1))

            return effective_weights

        try:
            # Check if the file exists and the size of data structures hasn't changed
            if not recompute_weights and os.path.exists(weights_file):
                with open(weights_file, 'rb') as f:
                    return pickle.load(f)
            else:
                effective_weights = precompute_effective_weights(document_weight_mappings=document_weight_mappings, authors_list=authors_list, authors_weights=authors_weights)
                with open(weights_file, 'wb') as f:
                    pickle.dump(effective_weights, f)
                return effective_weights
        except Exception as e:
            logging.error(f"Error while loading or computing weights: {str(e)}")
            return {}  # Return an empty dictionary in case of an error

    def penalise_if_discourse_only_or_forum_name_in_title(self, nodes_with_score: List[NodeWithScore]):
        # Check if DataFrames have required columns before using them
        merged_links = set()
        updated_titles = set()
        updated_links = set()

        if not self.merged_df.empty and 'Link' in self.merged_df.columns:
            merged_links = set(self.merged_df['Link'].dropna().unique())

        if not self.updated_df.empty:
            if 'title' in self.updated_df.columns:
                updated_titles = set(self.updated_df['title'].dropna().unique())
            if 'article' in self.updated_df.columns:
                updated_links = set(self.updated_df['article'].dropna().unique())

        for node_with_score in nodes_with_score:
            if node_with_score.node.metadata.get('document_type', '') == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                continue
            link = node_with_score.node.metadata.get('pdf_link', '').strip()
            title = node_with_score.node.metadata.get('title', '').strip()

            # Adjust score based on link and title conditions
            if link in merged_links and link not in updated_links:
                node_with_score.score *= self.discourse_only_penalty

            # Further adjust score if title contains specific keywords
            if "ethereum research" in title.lower() or "flashbots collective" in title.lower():
                node_with_score.score *= self.forum_name_in_title_penalty

        return nodes_with_score

    def populate_missing_pdf_links(self, nodes_with_score: List[NodeWithScore]):
        # Create a mapping from titles to links only if DataFrames are not empty
        titles_links_mapping = {}

        if not self.merged_df.empty and 'Title' in self.merged_df.columns and 'Link' in self.merged_df.columns:
            merged_mapping = self.merged_df[['Title', 'Link']].dropna().set_index('Title')['Link'].to_dict()
            titles_links_mapping.update(merged_mapping)

        if not self.updated_df.empty:
            # Check if required columns exist
            if 'article' in self.updated_df.columns and 'title' in self.updated_df.columns:
                updated_mapping = self.updated_df.rename(columns={'article': 'Link', 'title': 'Title'})[
                    ['Title', 'Link']].dropna().set_index('Title')['Link'].to_dict()
                titles_links_mapping.update(updated_mapping)

        for node_with_score in nodes_with_score:
            if node_with_score.node.metadata.get('document_type', '') == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                continue
            if not node_with_score.node.metadata.get('pdf_link'):
                title = node_with_score.node.metadata.get('title', '').strip()
                # Match the title and populate pdf_link if a corresponding link exists
                matched_link = titles_links_mapping.get(title)
                if matched_link:
                    node_with_score.node.metadata['pdf_link'] = matched_link

        return nodes_with_score

    def adjust_scores_based_on_criteria(self, nodes_with_score: List[NodeWithScore]):
        BOOST_SCORE_MULTIPLIER = float(os.environ.get('BOOST_SCORE_MULTIPLIER', '1.3'))
        DOCS_BOOST_SCORE_MULTIPLIER = float(os.environ.get('DOCS_BOOST_SCORE_MULTIPLIER', '1.20'))
        PENALTY_SCORE_MULTIPLIER = float(os.environ.get('PENALTY_SCORE_MULTIPLIER', '0.85'))
        CHANNEL_NAMES_TO_BOOST = [os.environ.get('CHANNEL_NAMES_TO_BOOST', 'ETHDenver')]  # Example channel names
        CHANNEL_NAMES_TO_PENALISE = [os.environ.get('CHANNEL_NAMES_TO_PENALISE', 'Chainlink')]  # NOTE 2024-03-19: chainlink is absolutely everywhere in most results, need to tune that down.
        DATE_THRESHOLD = datetime.strptime('2024-02-01', '%Y-%m-%d')

        for node_with_score in nodes_with_score:
            channel_name = node_with_score.node.metadata.get('channel_name', '').strip()
            if channel_name in CHANNEL_NAMES_TO_PENALISE:
                node_with_score.score *= PENALTY_SCORE_MULTIPLIER
            if channel_name in CHANNEL_NAMES_TO_BOOST:
                release_date_str = node_with_score.node.metadata.get('release_date', '').strip()
                try:
                    release_date = datetime.strptime(release_date_str, '%Y-%m-%d') if release_date_str else None
                except ValueError:
                    release_date = None

                if release_date and release_date > DATE_THRESHOLD:
                    node_with_score.score *= BOOST_SCORE_MULTIPLIER

            pdf_link = node_with_score.node.metadata.get('pdf_link', '').strip()
            if pdf_link == '':
                continue
            else:
                pdf_link_domain = urlparse(pdf_link).netloc
                if pdf_link_domain in self.site_domains or 'docs' in pdf_link.lower() or 'documentation' in pdf_link.lower():
                    node_with_score.score *= DOCS_BOOST_SCORE_MULTIPLIER
                    logging.info(f"Boosting score for document: [{node_with_score.node.metadata.get('title', 'UNSPECIFIED')}] from [{pdf_link}]")

        return nodes_with_score

    def apply_special_adjustments(self, node_with_score):
        document_name = node_with_score.node.metadata.get('title', 'UNSPECIFIED')

        # Special case adjustment
        if document_name in self.edge_case_set:
            logging.info(f"Applying [{self.doc_to_remove}] score to document: [{document_name}]")
            node_with_score.score *= self.doc_to_remove

        # Keyword penalisation
        for word in self.keywords_to_penalise:
            if word.lower() in document_name.lower():
                logging.info(f"Penalising keyword: [{word}] in document: [{document_name}]")
                node_with_score.score *= self.keyword_to_penalise_multiplier

    def nodes_reranker(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        NUM_CHUNKS_RETRIEVED = int(os.environ.get('NUM_CHUNKS_RETRIEVED', '10'))
        SCORE_THRESHOLD = float(os.environ.get('SCORE_THRESHOLD', '0.10'))
        MIN_CHUNKS_FOR_RESPONSE = int(os.environ.get('MIN_CHUNKS_FOR_RESPONSE', '2'))

        # Apply special case and keyword penalisation adjustments
        for node_with_score in nodes_with_score:
            self.apply_special_adjustments(node_with_score)

        logging.info(f"count nodes before score threshold filtering [{len(nodes_with_score)}]")
        # Filter out nodes below score threshold
        nodes_with_score = [node for node in nodes_with_score if node.score >= SCORE_THRESHOLD]
        logging.info(f"count nodes AFTER score threshold filtering [{len(nodes_with_score)}]")

        if len(nodes_with_score)==0:
            return nodes_with_score

        # Call the support method to check links and titles and adjust scores
        nodes_with_score = self.penalise_if_discourse_only_or_forum_name_in_title(nodes_with_score)

        # Populate missing pdf_link based on title matching
        nodes_with_score = self.populate_missing_pdf_links(nodes_with_score)

        # Apply scoring adjustments based on channel_name and release_date
        nodes_with_score = self.adjust_scores_based_on_criteria(nodes_with_score)

        if len(nodes_with_score) < MIN_CHUNKS_FOR_RESPONSE:
            logging.warning(f"Number of nodes below threshold: {len(nodes_with_score)}")
            nodes_with_score = []

        for node_with_score in nodes_with_score:
            score = node_with_score.score
            document_type = node_with_score.node.metadata.get('document_type', 'UNSPECIFIED').lower()
            document_name = node_with_score.node.metadata.get('title', 'UNSPECIFIED')

            if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
                source = node_with_score.node.metadata.get('channel_name', 'UNSPECIFIED LINK').strip()
                weight_key = (document_type + '_weights', source)
                effective_weight = self.effective_weights.get(
                    weight_key,
                    self.document_weights[document_type + '_weights'].get('default', 1)
                )
            else:
                link = node_with_score.node.metadata.get('pdf_link', 'UNSPECIFIED LINK').strip()
                extracted = tldextract.extract(link)
                domain = f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}".strip('.')
                author_url = node_with_score.node.metadata.get('authors', 'UNSPECIFIED LINK').strip()
                weight_key = (document_type, domain, author_url)  # + '_weights'
                effective_weight = self.effective_weights.get(
                    weight_key,
                    self.document_weights[document_type + '_weights'].get(domain, self.document_weights[document_type + '_weights'].get('default', 1))
                )

            score *= effective_weight
            node_with_score.score = score

        # Optional logging if in local environment
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            self.log_unique_filenames(nodes_with_score[:NUM_CHUNKS_RETRIEVED], f"Top {NUM_CHUNKS_RETRIEVED} nodes before rerank")

        # Get the top NUM_CHUNKS_RETRIEVED nodes based on score
        top_nodes = heapq.nlargest(NUM_CHUNKS_RETRIEVED, nodes_with_score, key=lambda x: x.score)
        top_nodes = [node for node in top_nodes if node.score >= SCORE_THRESHOLD]

        # Optional logging if in local environment
        if os.environ.get('ENVIRONMENT') == 'LOCAL':
            self.log_unique_filenames(top_nodes, f"Re-ranked top {NUM_CHUNKS_RETRIEVED} nodes")

        return top_nodes

    def adjust_score_for_document_type(self, document_type, authors, link, score, document_name):
        weight_key = document_type.lower() + '_weights'
        if weight_key in self.document_weights:
            source = authors if document_type == DOCUMENT_TYPES.YOUTUBE_VIDEO.value else link
            weight = self.find_weight_for_source(source, weight_key)
            score *= weight

            # Apply special case adjustment if applicable
            if document_name in self.edge_case_set:
                score *= 0.8

        return score

    def find_weight_for_source(self, source, weight_key):
        extracted = extract(source)

        # Join the subdomain and domain to get the key
        # e.g., 'blog.example.co.uk' will become 'blog.example'
        domain_key = '.'.join(filter(None, [extracted.subdomain, extracted.domain]))

        # Check if the domain_key is directly in the document_weight_mappings
        weight = self.document_weight_mappings.get(weight_key, {}).get(domain_key, self.default_weights.get(weight_key, 1))
        return weight

    def adjust_score_for_authors(self, authors, score, document_type):
        if document_type != DOCUMENT_TYPES.YOUTUBE_VIDEO.value:
            author_matched = False
            for author in authors.split(','):
                author = author.strip()
                if author in self.author_weight_mapping:
                    score *= self.author_weight_mapping[author]
                    author_matched = True
                    break  # Break the loop if an author matches

            if not author_matched:
                score *= self.author_weight_mapping['default']  # Apply default weight if no author matched

        return score

    def log_unique_filenames(self, nodes: List[NodeWithScore], context: str):
        unique_files_info = {}
        document_type_count = {}

        logging.info(f"Logging unique file names and document types in context: {context}")

        # Count occurrences of each title and collect scores
        title_scores = {}
        for node in nodes:
            title = node.node.metadata.get('title', 'UNSPECIFIED FILE')
            if title not in title_scores:
                title_scores[title] = []
            title_scores[title].append(node.score)

        for node in nodes:
            filename = node.node.metadata.get('title', 'UNSPECIFIED FILE')
            document_type = node.node.metadata.get('document_type', 'UNSPECIFIED')

            file_key = (document_type, filename)
            if file_key not in unique_files_info:
                unique_files_info[file_key] = {
                    'chunk_count': len(title_scores[filename]),
                    'index': len(unique_files_info) + 1,
                    'scores': title_scores[filename]  # Store the scores for this title
                }

            document_type_count[document_type] = document_type_count.get(document_type, 0) + 1

        for file_key, info in unique_files_info.items():
            document_type, filename = file_key
            scores_str = ", ".join(f"{score:.2f}" for score in info['scores'])  # Format scores as a comma-separated string
            logging.info(f"Unique file #{info['index']}: Document Type: [{document_type}], Filename: [{filename}], Chunks Retrieved: [{info['chunk_count']}], Scores: [{scores_str}]")

        # Constructing a single string for all document type counts
        document_type_counts_str = ", ".join(f"{doc_type}: {count}" for doc_type, count in document_type_count.items())
        logging.info(f"Document Type chunk-distribution: {document_type_counts_str}")

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
                CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                    CBEventType.RETRIEVE,
                    payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)

                # Log initial retrieval results
                logging.info(f"Initial retrieval returned {len(nodes)} nodes")
                if len(nodes) > 0:
                    logging.info(
                        f"Initial scores range: {min(n.score for n in nodes):.4f} - {max(n.score for n in nodes):.4f}")

                nodes = self.nodes_reranker(nodes_with_score=nodes)

                # Log after reranking
                logging.info(f"After reranking: {len(nodes)} nodes remain")

                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )

            if not nodes:
                response_str = """We could not find any results related to your query. 
                                   However, we encourage you to ask questions about Internet Capital Markets (ICM) and Solana, 
                                   as these topics are rich with information and ongoing developments. Feel free to ask another question!"""

                response = Response(
                    response_str,
                    source_nodes=[],
                    metadata={},
                )
            else:
                response = self._response_synthesizer.synthesize(
                    query=query_bundle,
                    nodes=nodes,
                )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
