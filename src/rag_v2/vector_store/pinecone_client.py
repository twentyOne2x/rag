from __future__ import annotations
import os
from typing import Optional
from pinecone import Pinecone

from ..settings import config_value

def get_pinecone_index(index_name: Optional[str] = None):
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    default_index = config_value("pinecone.index_name", default="icmfyi-v2")
    name = index_name or os.environ.get("PINECONE_INDEX_NAME", default_index)
    return pc.Index(name)
