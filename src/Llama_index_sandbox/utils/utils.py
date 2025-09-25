from __future__ import annotations
import csv
import time
import logging
import inspect
from datetime import datetime
from functools import wraps

from llama_index.legacy import OpenAIEmbedding

from llama_index.legacy.core.llms.types import ChatMessage, MessageRole


import os
import subprocess

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone


import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


# Markers that usually live at the project root
ROOT_MARKERS: tuple[str, ...] = (
    ".git",            # git repo
    "pyproject.toml",  # modern Python projects
    "setup.cfg",       # legacy packaging
    "src",             # your codebase layout hints at root
)


def _in_docker() -> bool:
    """Best-effort Docker detection."""
    if os.environ.get("IN_DOCKER", "").lower() in {"1", "true", "yes"}:
        return True
    return Path("/.dockerenv").exists()


def _scan_upward_for_root(start: Path, markers: Iterable[str], max_levels: int = 12) -> Optional[Path]:
    """Walk up from `start` looking for any of `markers`."""
    cur = start.resolve()
    root = cur.anchor  # filesystem root (e.g., "/")
    for _ in range(max_levels):
        if any((cur / m).exists() for m in markers):
            return cur
        if str(cur) == root:
            break
        cur = cur.parent
    return None


def _git_root(start: Path) -> Optional[Path]:
    """Return git repo root if available, else None (never raises)."""
    if not shutil.which("git"):
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
        return Path(out.decode("utf-8", "replace").strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def root_directory(start_from: Optional[Path | str] = None) -> str:
    """
    Resolve the project root once and cache it.

    Resolution order:
      1) PROJECT_ROOT env var (if set and exists)
      2) Docker default '/app' (if detected and exists)
      3) git rev-parse --show-toplevel (if git present)
      4) Upward scan for common project markers
    """
    # 1) Env override
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return str(p)

    # 2) Docker default
    if _in_docker():
        p = Path("/app")
        if p.exists():
            return str(p)

    # Determine a reasonable starting point
    start = Path(start_from) if start_from else Path.cwd()

    # 3) Git root
    git_root = _git_root(start)
    if git_root and git_root.exists():
        return str(git_root)

    # 4) Upward scan for markers
    scanned = _scan_upward_for_root(start, ROOT_MARKERS, max_levels=16)
    if scanned:
        return str(scanned)

    # Last resort: try where this file lives (useful if imported from elsewhere)
    here = Path(__file__).resolve().parent
    scanned_from_here = _scan_upward_for_root(here, ROOT_MARKERS, max_levels=16)
    if scanned_from_here:
        return str(scanned_from_here)

    raise RuntimeError(
        "Could not determine project root. Set the PROJECT_ROOT environment variable to override."
    )

root_dir=root_directory()


from src.Llama_index_sandbox.utils.logging_setup import add_section_file_logger
from datetime import datetime
import os, logging

def start_logging(log_prefix: str):
    logs_dir = os.path.join(root_dir, "logs", "txt")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(logs_dir, f"{ts}_{log_prefix}.log")
    add_section_file_logger(path)
    logging.info(f"********* {log_prefix} LOGGING STARTED *********")

def timeit(func):
    """
    A decorator that logs the time a function takes to execute along with the directory and filename.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to execute the decorated function and log its execution time and location.

        Args:
            *args: Variable length argument list to pass to the decorated function.
            **kwargs: Arbitrary keyword arguments to pass to the decorated function.

        Returns:
            The value returned by the decorated function.
        """
        if os.getenv('ENVIRONMENT') == 'LOCAL':
            # Get the current file's path and extract directory and filename
            file_path = inspect.getfile(func)
            directory, filename = os.path.split(file_path)
            dir_name = os.path.basename(directory)

            # Log start of function execution
            logging.info(f"{dir_name}.{filename}.{func.__name__} STARTED.")
            start_time = time.time()

            # Call the decorated function and store its result
            result = func(*args, **kwargs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            # Log end of function execution
            logging.info(f"{dir_name}.{filename}.{func.__name__} COMPLETED, took {int(minutes)} minutes and {seconds:.2f} seconds to run.\n")

            return result
        else:
            # If not in 'LOCAL' environment, just call the function without timing
            return func(*args, **kwargs)

    return wrapper

def get_last_index_embedding_params():
    index_dir = f"{root_dir}/.storage/research_pdf/"
    index = sorted(os.listdir(index_dir))[-1].split('_')
    index_date = index[0]
    embedding_model_name = index[1]
    embedding_model_chunk_size = int(index[2])
    chunk_overlap = int(index[3])
    vector_space_distance_metric = 'cosine'  # TODO 2023-11-02: save vector_space_distance_metric in index name
    return embedding_model_name, embedding_model_chunk_size, chunk_overlap, vector_space_distance_metric



def fullwidth_to_ascii(char):
    """Converts a full-width character to its ASCII equivalent."""
    # Full-width range: 0xFF01-0xFF5E
    # Corresponding ASCII range: 0x21-0x7E
    fullwidth_offset = 0xFF01 - 0x21
    return chr(ord(char) - fullwidth_offset) if 0xFF01 <= ord(char) <= 0xFF5E else char


def process_messages(data):
    try:
        messages = data["chat_history"]
    except KeyError:
        # Handle the absence of chat_history key more gracefully
        return None
    chat_messages = []

    for message in messages:
        # Create a ChatMessage object for each message
        chat_message = ChatMessage(
            role=MessageRole(message.get("role", "user").lower()),  # Convert the role to Enum
            content=message.get("content", ""),
            additional_kwargs=message.get("additional_kwargs", {})  # Assuming additional_kwargs is part of your message structure
        )
        chat_messages.append(chat_message)

    return chat_messages


def get_embedding_model(embedding_model_name):
    if embedding_model_name == "text-embedding-3-large":
        embedding_model = OpenAIEmbedding(
            model="text-embedding-3-large",  # This model produces 3072-dimensional vectors
            dimensions=3072  # Explicitly set dimensions
        )
    else:
        assert False, f"The embedding model is not supported: [{embedding_model_name}]"
    return embedding_model


def load_csv_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        logging.warning(f"CSV file not found at path: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file doesn't exist


def load_vector_store_from_pinecone_database(delete_old_index=False, new_index=False, index_name=os.environ.get("PINECONE_INDEX_NAME", "icmfyi")):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    if new_index:
        # pass
        if delete_old_index:
            logging.warning(f"Are you sure you want to delete the old index with name [{index_name}]?")
            pc.delete_index(index_name)
        # Dimensions are for text-embedding-ada-002
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


def load_vector_store_from_pinecone_database_legacy(index_name=os.environ.get("PINECONE_INDEX_NAME", "icmfyi")):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    pinecone_index = pc.Index(index_name)
    # from llama_index.legacy.vector_stores import PineconeVectorStore
    import llama_index.legacy.vector_stores as legacy_vector_stores

    vector_store = legacy_vector_stores.PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


if __name__ == '__main__':
    pass
