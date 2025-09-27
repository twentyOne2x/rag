# https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
# Credits to https://gpt-index.readthedocs.io/en/stable/examples/low_level/ingestion.html
import logging
import os
from src.Llama_index_sandbox.utils.logging_setup import configure_logging
from src.Llama_index_sandbox import root_dir

# — core Settings (no ServiceContext) —
from llama_index.core import Settings

import src.Llama_index_sandbox.utils.utils
from src.Llama_index_sandbox import config_instance
from src.Llama_index_sandbox.constants import INPUT_QUERIES
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_query_engine import CustomQueryEngine
from src.Llama_index_sandbox.utils.gcs_utils import set_secrets_from_cloud
from src.Llama_index_sandbox.retrieve import (
    get_engine_from_vector_store,
    ask_questions,
    get_inference_llm,
)
from src.Llama_index_sandbox.utils.utils import start_logging, get_last_index_embedding_params
from src.Llama_index_sandbox.index import load_index_from_disk


def initialise_chatbot(engine: str, query_engine_as_tool: bool):
    stream = True
    # read run config
    similarity_top_k = config_instance.NUM_CHUNKS_SEARCHED_FOR_RERANKING[0]
    text_splitter_chunk_size = config_instance.CHUNK_SIZES[0]
    text_splitter_chunk_overlap_percentage = config_instance.CHUNK_OVERLAPS[0]

    # models
    embedding_model_name = config_instance.EMBEDDING_MODELS[0]
    embedding_model = src.Llama_index_sandbox.utils.utils.get_embedding_model(
        embedding_model_name=embedding_model_name
    )

    llm_model_name = config_instance.INFERENCE_MODELS[0]
    llm = get_inference_llm(llm_model_name=llm_model_name)

    # --- Settings bootstrap (replaces ServiceContext) ---
    Settings.llm = llm
    from llama_index.embeddings.openai import OpenAIEmbedding

    # MUST match the index dim
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        dimensions=3072,  # <- crucial
    )

    start_logging(
        f"create_index_{embedding_model_name.split('/')[-1]}_{llm_model_name}_"
        f"{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}_{similarity_top_k}"
    )
    (
        index_embedding_model_name,
        index_text_splitter_chunk_size,
        index_chunk_overlap,
        vector_space_distance_metric,
    ) = get_last_index_embedding_params()
    logging.info(
        "index_embedding_model_name=%s, index_text_splitter_chunk_size=%s, "
        "index_chunk_overlap=%s, vector_space_distance_metric=%s",
        index_embedding_model_name,
        index_text_splitter_chunk_size,
        index_chunk_overlap,
        vector_space_distance_metric,
    )

    # load index (now relies on global Settings; no args)
    index = load_index_from_disk()

    # Retrieve/Query from the Vector Store
    log_name = (
        f"{embedding_model_name.split('/')[-1]}_{llm_model_name}_"
        f"{text_splitter_chunk_size}_{text_splitter_chunk_overlap_percentage}"
    )
    start_logging(f"ask_questions_{log_name}_{similarity_top_k}")

    # NOTE: get_engine_from_vector_store no longer takes service_context
    retrieval_engine, query_engine, store_response_partial = get_engine_from_vector_store(
        embedding_model_name=embedding_model_name,
        embedding_model=embedding_model,
        llm_model_name=llm_model_name,
        text_splitter_chunk_size=text_splitter_chunk_size,
        text_splitter_chunk_overlap_percentage=text_splitter_chunk_overlap_percentage,
        similarity_top_k=similarity_top_k,
        index=index,
        engine=engine,
        stream=stream,
        query_engine_as_tool=query_engine_as_tool,
        log_name=log_name,
    )

    return retrieval_engine, query_engine, store_response_partial, config_instance


def run():
    log_path = os.path.join(root_dir, "logs", "run.log")
    configure_logging(log_path)

    if os.environ.get("ENVIRONMENT") != "LOCAL":
        set_secrets_from_cloud()

    # (Re)compute weights (safe if file exists)
    CustomQueryEngine.load_or_compute_weights(
        document_weight_mappings=CustomQueryEngine.document_weight_mappings,
        weights_file=CustomQueryEngine.weights_file,
        authors_list=CustomQueryEngine.authors_list,
        authors_weights=CustomQueryEngine.authors_weights,
        recompute_weights=True,
    )

    engine = "chat"
    query_engine_as_tool = True
    chat_history = []

    logging.info("Run parameters: engine=%s, query_engine_as_tool=%s", engine, query_engine_as_tool)

    retrieval_engine, query_engine, store_response_partial, cfg = initialise_chatbot(
        engine=engine, query_engine_as_tool=query_engine_as_tool
    )

    ask_questions(
        input_queries=INPUT_QUERIES[:2],
        retrieval_engine=retrieval_engine,
        query_engine=query_engine,
        store_response_partial=store_response_partial,
        engine=engine,
        query_engine_as_tool=query_engine_as_tool,
        chat_history=chat_history,
        reset_chat=cfg.reset_chat,
    )
    return retrieval_engine


if __name__ == "__main__":
    run()
