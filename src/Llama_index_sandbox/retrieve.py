import logging
import os
from functools import partial
from typing import Optional, Type, Union

# ---- Core-first imports (no ServiceContext) ----
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.callbacks import CallbackManager

# Chat/query engine base types
try:
    from llama_index.core.chat_engine.types import BaseChatEngine
except ImportError:
    class BaseChatEngine:  # type: ignore
        pass

from llama_index.llms.openai import OpenAI

# Core LLM types
from llama_index.core.llms import ChatMessage, MessageRole

# Core memory
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer

from llama_index.core.utils import print_text

from src.Llama_index_sandbox.constants import (
    LLM_TEMPERATURE,
)
from src.Llama_index_sandbox.custom_react_agent.logging_handler import JSONLoggingHandler
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_vector_store_index import CustomVectorStoreIndex
from src.Llama_index_sandbox.prompts import (
    QUERY_TOOL_RESPONSE,
    QUERY_ENGINE_TOOL_DESCRIPTION,
)
from src.Llama_index_sandbox.custom_react_agent.ReActAgent import CustomReActAgent
from src.Llama_index_sandbox.custom_react_agent.formatter import CustomReActChatFormatter
from src.Llama_index_sandbox.custom_react_agent.output_parser import CustomReActOutputParser
from src.Llama_index_sandbox.custom_react_agent.tools.fn_schema import ToolFnSchema

from src.Llama_index_sandbox.custom_react_agent.tools.query_engine import CustomQueryEngineTool
from src.Llama_index_sandbox.custom_react_agent.tools.tool_output import log_and_store
from src.Llama_index_sandbox.utils.store_response import store_response
from src.Llama_index_sandbox.utils.utils import timeit
from src.Llama_index_sandbox.utils.text_sanitizer import strip_meta_phrases


# ----------------------- Query Engine Helpers -----------------------

def get_query_engine(index, verbose: bool = True, similarity_top_k: int = 5):
    """Create a query engine with a compact response synthesizer (core APIs)."""
    from llama_index.core.response_synthesizers import (
        get_response_synthesizer,
        ResponseMode,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        verbose=verbose,
    )
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        verbose=verbose,
        response_synthesizer=response_synthesizer,
    )


def get_inference_llm(llm_model_name: str):
    """
    Return a core OpenAI LLM instance. No legacy remaps.
    Let model availability errors surface, or handle upstream.
    """
    return OpenAI(model=llm_model_name, callback_manager=CallbackManager([]))


def set_inference_llm_params(
    temperature: float,
    stream: bool = False,
    callback_manager: Optional[CallbackManager] = None,
    max_tokens: Optional[int] = None,
    llm: Optional[OpenAI] = None,
):
    """
    Configure the LLM in-place. Defaults to Settings.llm if 'llm' is not passed.
    """
    llm = llm or Settings.llm
    if hasattr(llm, "temperature"):
        llm.temperature = temperature
    if callback_manager is not None and hasattr(llm, "callback_manager"):
        llm.callback_manager = callback_manager
    if max_tokens is not None and hasattr(llm, "max_tokens"):
        llm.max_tokens = max_tokens
    # If you adopt a streaming path later, wire it here (varies by LLM impl).
    return llm


# ----------------------- Chat Engine Factory -----------------------

def get_chat_engine(
    index: CustomVectorStoreIndex,
    query_engine_as_tool: bool,
    stream: bool,
    log_name: str,
    chat_mode: str = "react",
    verbose: bool = True,
    similarity_top_k: int = 5,
    max_iterations: int = 10,
    memory: Optional[BaseMemory] = None,
    memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
    temperature: float = LLM_TEMPERATURE,
):
    """
    Builds your ReAct-style chat engine, wiring the query engine as a tool when requested.
    Uses global Settings for LLM/embeddings (no ServiceContext).
    """
    logging.info(f"Fetching query engine tool from index object # [{index}]")
    query_engine = get_query_engine(index=index, verbose=verbose, similarity_top_k=similarity_top_k)
    logging.info("Successfully instantiated the query engine!")

    query_engine_tool = CustomQueryEngineTool.from_defaults(query_engine=query_engine)
    query_engine_tool.metadata.description = QUERY_ENGINE_TOOL_DESCRIPTION
    query_engine_tool.metadata.fn_schema = ToolFnSchema

    react_chat_formatter = CustomReActChatFormatter(tools=[query_engine_tool])
    output_parser = CustomReActOutputParser()

    # JSON logging
    json_logging_handler = JSONLoggingHandler(
        event_ends_to_ignore=[],
        event_starts_to_ignore=[],
        log_name=log_name,
        similarity_top_k=similarity_top_k,
    )
    callback_manager = CallbackManager(handlers=[json_logging_handler])

    # Cap tokens to control costs unless explicitly overridden
    max_tokens_env = os.environ.get("LLM_MAX_TOKENS", "600")
    try:
        max_tokens_cap = int(max_tokens_env)
    except ValueError:
        max_tokens_cap = 600

    # Configure LLM in Settings
    llm = set_inference_llm_params(
        temperature=temperature,
        stream=stream,
        callback_manager=callback_manager,
        max_tokens=max_tokens_cap,
        llm=Settings.llm,
    )

    # Memory
    memory = memory or memory_cls.from_defaults(chat_history=[], llm=llm)

    # ⚠️ Important: directly construct the agent; do NOT call `.from_tools`
    agent_kwargs = dict(
        tools=[query_engine_tool] if query_engine_as_tool else [],
        react_chat_formatter=react_chat_formatter,
        llm=llm,
        max_iterations=max_iterations,
        memory=memory,
        output_parser=output_parser,
        verbose=verbose,
    )

    try:
        # Most builds accept constructor kwargs like below
        return CustomReActAgent(**agent_kwargs)
    except TypeError:
        # Fallback shape if constructor signature differs
        # (older legacy ReActAgent often takes these positionally)
        return CustomReActAgent(
            tools=agent_kwargs["tools"],
            react_chat_formatter=react_chat_formatter,
            llm=llm,
            max_iterations=max_iterations,
            memory=memory,
            output_parser=output_parser,
            verbose=verbose,
        )


# ----------------------- Chat / Query Execution -----------------------

def ask_questions(
    input_queries,
    retrieval_engine,
    query_engine,
    store_response_partial,
    engine,
    query_engine_as_tool,
    reset_chat,
    chat_history,
    direct_llm_call: bool = False,
    run_application: bool = False,
):
    """
    Runs queries through either the chat agent (has .chat) or a direct query engine (has .query),
    logs/stores responses, and returns the last response + metadata when applicable.
    """
    all_formatted_metadata = None

    # Duck-typed capability checks
    has_chat = hasattr(retrieval_engine, "chat") and callable(getattr(retrieval_engine, "chat"))
    has_query = hasattr(retrieval_engine, "query") and callable(getattr(retrieval_engine, "query"))

    for query_str in input_queries:
        if has_chat:
            if not query_engine_as_tool:
                # Use query_engine directly first, then pass to chat agent
                response = query_engine.query(query_str)
                str_response, all_formatted_metadata = log_and_store(
                    store_response_partial, query_str, response, chatbot=True
                )
                str_response = QUERY_TOOL_RESPONSE.format(question=query_str, response=str_response)
                logging.info(f"Message passed to chat engine:    \n\n[{str_response}]")
                logging.info(f"With input chat history: [{chat_history}]")
                response, all_formatted_metadata = retrieval_engine.chat(
                    message=str_response, chat_history=chat_history
                )
            else:
                if os.environ.get("ENVIRONMENT") == "LOCAL":
                    logging.info(f"The question asked is: [{query_str}]")
                    logging.info(f"With input chat history: [{chat_history}]")
                if not direct_llm_call:
                    response, all_formatted_metadata = retrieval_engine.chat(
                        message=query_str, chat_history=chat_history
                    )
                else:
                    # Best-effort direct LLM call; relies on your agent having _llm
                    response, all_formatted_metadata = retrieval_engine._llm.chat(
                        [ChatMessage(content=query_str, role="user")]
                    )

            text_out = getattr(response, "response", response)
            text_out = strip_meta_phrases(text_out)

            if not run_application:
                logging.info(
                    f"[End output shown to client for question [{query_str}]]:    \n```\n{text_out}\n```"
                )
                if os.environ.get("ENVIRONMENT") == "LOCAL":
                    print_text(
                        f"[End output shown to client for question [{query_str}]]:    \n```\n{text_out}\n\n Fetched based on the following sources: \n{all_formatted_metadata}\n```\n",
                        color="green",
                    )
            if reset_chat and hasattr(retrieval_engine, "reset"):
                logging.info("Resetting chat engine after question.")
                retrieval_engine.reset()

        elif has_query:
            logging.info(f"Querying index with query:    [{query_str}]")
            response = retrieval_engine.query(query_str)
            response, all_formatted_metadata = log_and_store(
                store_response_partial, query_str, response, chatbot=False
            )

        else:
            # Neither chat nor query method is present — surface a clear error
            raise TypeError(
                f"Retrieval engine has neither .chat nor .query. Got type={type(retrieval_engine)}"
            )

    if (len(input_queries) == 1) or (all_formatted_metadata is not None):
        return response, all_formatted_metadata


# ----------------------- Engine Wiring -----------------------

@timeit
def get_engine_from_vector_store(
    embedding_model_name: str,
    embedding_model: Union[OpenAIEmbedding, HuggingFaceEmbedding],
    llm_model_name: str,
    text_splitter_chunk_size: int,
    text_splitter_chunk_overlap_percentage: int,
    index: CustomVectorStoreIndex,
    query_engine_as_tool: bool,
    stream: bool,
    similarity_top_k: int,
    log_name: str,
    engine: str = "chat",
):
    """
    Creates the retrieval/chat engines from a vector store-backed index.
    No ServiceContext; relies on global Settings for LLM/embeddings.
    """
    # Partial for logging/storing responses
    store_response_partial = partial(
        store_response,
        embedding_model_name,
        llm_model_name,
        text_splitter_chunk_size,
        text_splitter_chunk_overlap_percentage,
    )

    if engine == "chat":
        retrieval_engine = get_chat_engine(
            index=index,
            stream=stream,
            query_engine_as_tool=query_engine_as_tool,
            log_name=log_name,
            verbose=True,
            similarity_top_k=similarity_top_k,
        )
        query_engine = get_query_engine(index=index, verbose=True, similarity_top_k=similarity_top_k)
    elif engine == "query":
        query_engine = None
        retrieval_engine = get_query_engine(index=index, verbose=True, similarity_top_k=similarity_top_k)
    else:
        assert False, f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}"

        # TODO 2023-10-05 [RETRIEVAL]: in particular for chunks from youtube videos, we might want
        #   to expand the window from which it retrieved the chunk
        # TODO 2023-10-05 [RETRIEVAL]: since many chunks can be retrieved from a single youtube video,
        #   what should be the returned timestamp to these references? should we return them all? return the one with highest score?
        # TODO 2023-10-05 [RETRIEVAL]: add weights such that responses from older sources have less importance in the answer
        # TODO 2023-10-05 [RETRIEVAL]: should we weight more a person which is an author and has a paper?
        # TODO 2023-10-07 [RETRIEVAL]: ADD metadata filtering e.g. "only video" or "only papers", or "from this author", or "from this channel", or "from 2022 and 2023" etc
        # TODO 2023-10-07 [RETRIEVAL]: in the chat format, is the rag system keeping in memory the previous retrieved chunks? e.g. if an answer is too short can it develop it further?
        # TODO 2023-10-07 [RETRIEVAL]: should we allow the external user to tune the top-k retrieved chunks?

        # TODO 2023-10-09 [RETRIEVAL]: use metadata tags for users to choose amongst LVR, Intents, MEV, etc such that it can increase the result speed (and likely accuracy)
        #  and this upfront work is likely a low hanging fruit relative to payoff.

    return retrieval_engine, query_engine, store_response_partial
