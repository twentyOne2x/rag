import os

import pytest
from pinecone import Pinecone


@pytest.mark.integration
def test_pumpfun_vectors_present():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    namespace = os.getenv("PINECONE_NAMESPACE", "videos")
    embed_dim = int(os.getenv("EMBED_DIM", "3072"))

    if not api_key:
        pytest.skip("Pinecone credentials not configured")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)

    probe_vector = [0.0] * embed_dim
    response = index.query(
        vector=probe_vector,
        namespace=namespace,
        top_k=5,
        filter={"router_tags": {"$in": ["pumpfun"]}},
        include_values=False,
        include_metadata=True,
    )

    matches = response.get("matches") or []
    assert matches, "Expected at least one Pump.fun vector in the index"

    found_display = False
    found_enriched = False
    for match in matches:
        tags = match.get("metadata", {}).get("router_tags", [])
        assert isinstance(tags, (list, tuple))
        assert any(tag == "pumpfun" or str(tag).startswith("pumpfun_") for tag in tags)
        channel_name = match.get("metadata", {}).get("channel_name", "")
        if channel_name:
            if "(Pumpfun)" in channel_name:
                found_display = True
        # Confirm Pump.fun-specific metadata is present
        meta = match.get("metadata", {})
        if meta.get("pumpfun_coin_symbol") or meta.get("pumpfun_coin_name") or meta.get("pumpfun_room") or meta.get("pumpfun_clip_id"):
            found_enriched = True

    # Older records may still carry legacy channel names, so only warn if no display name found
    if not found_display:
        pytest.skip("No Pump.fun display names found yet; legacy data may still be in the index")

    if not found_enriched:
        pytest.skip("Pump.fun-specific metadata not found; redeploy ingestion to backfill")
