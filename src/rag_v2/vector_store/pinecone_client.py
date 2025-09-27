from __future__ import annotations
import os
from typing import Optional
from pinecone import Pinecone

def get_pinecone_index(index_name: Optional[str] = None):
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name or os.environ.get("PINECONE_INDEX_NAME","icmfyi-v2"))
    return idx
