from __future__ import annotations
import csv
import time
import logging
import inspect
from datetime import datetime
from functools import wraps, lru_cache
from pathlib import Path
from typing import Iterable, Optional

import os
import shutil
import subprocess

import pandas as pd  # <-- missing before

# --- LlamaIndex (CORE) imports: no legacy here ---
from llama_index.embeddings.openai import OpenAIEmbedding  # CORE embedding
from llama_index.core.llms import ChatMessage, MessageRole  # CORE message types

# Pinecone (core vector store path)
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

# ------------------------- Project root helpers -------------------------

ROOT_MARKERS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "setup.cfg",
    "src",
)

def _in_docker() -> bool:
    if os.environ.get("IN_DOCKER", "").lower() in {"1", "true", "yes"}:
        return True
    return Path("/.dockerenv").exists()

def _scan_upward_for_root(start: Path, markers: Iterable[str], max_levels: int = 12) -> Optional[Path]:
    cur = start.resolve()
    root = cur.anchor
    for _ in range(max_levels):
        if any((cur / m).exists() for m in markers):
            return cur
        if str(cur) == root:
            break
        cur = cur.parent
    return None

def _git_root(start: Path) -> Optional[Path]:
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
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return str(p)

    if _in_docker():
        p = Path("/app")
        if p.exists():
            return str(p)

    start = Path(start_from) if start_from else Path.cwd()

    git_root = _git_root(start)
    if git_root and git_root.exists():
        return str(git_root)

    scanned = _scan_upward_for_root(start, ROOT_MARKERS, max_levels=16)
    if scanned:
        return str(scanned)

    here = Path(__file__).resolve().parent
    scanned_from_here = _scan_upward_for_root(here, ROOT_MARKERS, max_levels=16)
    if scanned_from_here:
        return str(scanned_from_here)

    raise RuntimeError(
        "Could not determine project root. Set PROJECT_ROOT environment variable to override."
    )

root_dir = root_directory()

# ------------------------- Logging helpers -------------------------

from src.Llama_index_sandbox.utils.logging_setup import add_section_file_logger

def start_logging(log_prefix: str):
    logs_dir = os.path.join(root_dir, "logs", "txt")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(logs_dir, f"{ts}_{log_prefix}.log")
    add_section_file_logger(path)
    logging.info(f"********* {log_prefix} LOGGING STARTED *********")

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv("ENVIRONMENT") == "LOCAL":
            file_path = inspect.getfile(func)
            directory, filename = os.path.split(file_path)
            dir_name = os.path.basename(directory)

            logging.info(f"{dir_name}.{filename}.{func.__name__} STARTED.")
            start_time = time.time()

            result = func(*args, **kwargs)

            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            logging.info(f"{dir_name}.{filename}.{func.__name__} COMPLETED, took {int(minutes)} minutes and {seconds:.2f} seconds.\n")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

# ------------------------- Index params -------------------------

def get_last_index_embedding_params():
    """Return the metadata encoded in the most recent index folder.

    When the persisted index assets are unavailable (e.g. in a fresh Cloud Run
    container) fall back to sensible defaults driven by environment variables
    so the service can still boot.
    """
    index_dir = Path(root_dir) / ".storage" / "research_pdf"
    fallback_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
    fallback_chunk_size = int(os.getenv("DEFAULT_TEXT_SPLITTER_CHUNK_SIZE", "750"))
    fallback_chunk_overlap = int(os.getenv("DEFAULT_TEXT_SPLITTER_CHUNK_OVERLAP", "10"))
    vector_space_distance_metric = "cosine"

    try:
        entries = sorted(p for p in index_dir.iterdir() if p.is_dir())
    except FileNotFoundError:
        logging.warning("Index directory %s not found; using fallback embedding parameters.", index_dir)
        return fallback_model, fallback_chunk_size, fallback_chunk_overlap, vector_space_distance_metric

    if not entries:
        logging.warning("Index directory %s is empty; using fallback embedding parameters.", index_dir)
        return fallback_model, fallback_chunk_size, fallback_chunk_overlap, vector_space_distance_metric

    latest = entries[-1].name.split("_")
    if len(latest) < 4:
        logging.warning("Malformed index folder name '%s'; using fallback embedding parameters.", entries[-1].name)
        return fallback_model, fallback_chunk_size, fallback_chunk_overlap, vector_space_distance_metric

    embedding_model_name = latest[1]
    embedding_model_chunk_size = int(latest[2])
    chunk_overlap = int(latest[3])
    return embedding_model_name, embedding_model_chunk_size, chunk_overlap, vector_space_distance_metric

# ------------------------- Chat utils -------------------------

def fullwidth_to_ascii(char):
    fullwidth_offset = 0xFF01 - 0x21
    return chr(ord(char) - fullwidth_offset) if 0xFF01 <= ord(char) <= 0xFF5E else char

def _normalize_role(role_str: str) -> MessageRole:
    role = (role_str or "user").strip().lower()
    if role == "assistant":
        return MessageRole.ASSISTANT
    if role == "system":
        return MessageRole.SYSTEM
    return MessageRole.USER

def process_messages(data):
    messages = data.get("chat_history")
    if not messages:
        return None
    chat_messages = []
    for message in messages:
        chat_messages.append(
            ChatMessage(
                role=_normalize_role(message.get("role", "user")),
                content=message.get("content", ""),
                additional_kwargs=message.get("additional_kwargs", {}),
            )
        )
    return chat_messages

# ------------------------- Embeddings (CORE) -------------------------

def get_embedding_model(embedding_model_name: str):
    """
    Return a CORE OpenAIEmbedding instance (not legacy).
    Do NOT pass legacy embeddings into Settings.embed_model.
    """
    if embedding_model_name == "text-embedding-3-large":
        # 3072 dims by default; no need to pass dimensions explicitly
        return OpenAIEmbedding(model="text-embedding-3-large")
    else:
        raise AssertionError(f"Unsupported embedding model: [{embedding_model_name}]")

# ------------------------- CSV helper -------------------------

def load_csv_data(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    logging.warning(f"CSV file not found at path: {file_path}")
    return pd.DataFrame()

# ------------------------- Pinecone vector store -------------------------

def load_vector_store_from_pinecone_database(
    delete_old_index: bool = False,
    new_index: bool = False,
    index_name: str = os.environ.get("PINECONE_INDEX_NAME", "icmfyi-v2"),
):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if new_index:
        if delete_old_index:
            logging.warning(f"Deleting old index: [{index_name}]")
            pc.delete_index(index_name)
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )

    pinecone_index = pc.Index(index_name)
    return PineconeVectorStore(pinecone_index=pinecone_index)

def load_vector_store_from_pinecone_database_legacy(
    index_name: str = os.environ.get("PINECONE_INDEX_NAME", "icmfyi-v2")
):
    """
    Keep this only if you still rely on legacy paths elsewhere.
    Prefer the core PineconeVectorStore above.
    """
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index(index_name)

    # legacy vector store kept for backward compatibility during migration
    import llama_index.legacy.vector_stores as legacy_vector_stores
    return legacy_vector_stores.PineconeVectorStore(pinecone_index=pinecone_index)

# -------------------------

if __name__ == "__main__":
    pass
