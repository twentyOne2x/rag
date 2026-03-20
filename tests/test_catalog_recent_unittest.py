import unittest
from unittest.mock import patch

from src.rag_v2.vector_store.parent_catalog import (
    CatalogRow,
    decode_recent_cursor,
    encode_recent_cursor,
    list_recent_parent_catalog,
)


def _row(*, vid: str, pub: str | None) -> CatalogRow:
    return CatalogRow(
        video_id=vid,
        title=f"t-{vid}",
        description=None,
        channel_name="ch",
        channel_id="cid",
        published_at=pub,
        duration_s=1.0,
        url=f"https://example.com/{vid}",
        thumbnail_url=None,
        source="youtube",
        document_type="youtube_video",
        topic_summary=None,
        router_tags=["ai"],
        aliases=None,
        canonical_entities=["Some Person"],
        speaker_names=None,
        entities=["Some Person"],
        router_boost=None,
    )


class TestCatalogRecent(unittest.TestCase):
    def test_cursor_roundtrip(self):
        c = encode_recent_cursor(published_at="2026-02-15", video_id="abc123")
        k = decode_recent_cursor(c)
        self.assertEqual(k, ("2026-02-15", "abc123"))

    def test_recent_pagination_and_since(self):
        rows = [
            _row(vid="v1", pub="2026-02-15"),
            _row(vid="v2", pub="2026-02-14"),
            _row(vid="v3", pub="2026-02-13"),
        ]

        def fake_load_parent_catalog(*, namespace: str, force_refresh: bool = False):
            self.assertEqual(namespace, "videos")
            return rows

        with patch("src.rag_v2.vector_store.parent_catalog.load_parent_catalog", new=fake_load_parent_catalog):
            page1 = list_recent_parent_catalog(namespace="videos", limit=2, since="2026-02-14")
            self.assertEqual(page1["returned"], 2)
            self.assertEqual([r["video_id"] for r in page1["results"]], ["v1", "v2"])
            self.assertFalse(page1["exhausted"])
            self.assertTrue(page1["next_cursor"])

            page2 = list_recent_parent_catalog(
                namespace="videos", limit=2, since="2026-02-14", cursor=page1["next_cursor"]
            )
            self.assertEqual(page2["returned"], 0)
            self.assertTrue(page2["exhausted"])


if __name__ == "__main__":
    unittest.main()

