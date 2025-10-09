from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal

DocType = Literal["youtube_video","stream"]

class ParentNode(BaseModel):
    node_type: Literal["parent"] = "parent"
    parent_id: str
    document_type: DocType
    title: str
    description: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    speaker_primary: Optional[str] = None
    published_at: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    duration_s: float = 0
    url: HttpUrl
    language: Optional[str] = "en"
    entities: List[str] = []
    chapters: Optional[list] = None
    ingest_version: int = 2
    # router extras
    router_tags: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    canonical_entities: Optional[List[str]] = None
    is_explainer: Optional[bool] = None
    router_boost: Optional[float] = None
    topic_summary: Optional[str] = None

class ChildNode(BaseModel):
    node_type: Literal["child"] = "child"
    segment_id: str
    parent_id: str
    document_type: DocType
    text: str
    start_s: float
    end_s: float
    start_hms: str
    end_hms: str
    clip_url: Optional[HttpUrl] = None
    speaker: Optional[str] = None
    channel_name: Optional[str] = None
    channel_id: Optional[str] = None
    entities: List[str] = []
    chapter: Optional[str] = None
    language: Optional[str] = "en"
    ingest_version: int = 2
