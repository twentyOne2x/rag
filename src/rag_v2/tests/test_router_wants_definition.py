from __future__ import annotations

from rag_v2.router.video_router import wants_definition


def test_wants_definition_matches_what_does_mean() -> None:
    assert wants_definition("What does SIMD mean?")


def test_wants_definition_matches_acronym_only() -> None:
    assert wants_definition("SIMD")
    assert wants_definition("SIMD?")


def test_wants_definition_matches_define() -> None:
    assert wants_definition("Define SIMD")
