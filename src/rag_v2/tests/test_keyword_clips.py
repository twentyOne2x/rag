from rag_v2.vector_store.keyword_clips import build_keyword_matcher


def test_keyword_matcher_single_token_matches_whole_word_only():
    m = build_keyword_matcher(query="credit")
    assert m.matches("We talk about Credit today.")
    assert m.matches("credit is important")
    assert not m.matches("accreditation is a different word")
    assert not m.matches("creditor risk")  # substring should not match


def test_keyword_matcher_phrase_matches_punctuation_and_hyphen_variants():
    m = build_keyword_matcher(query="onchain credit")
    assert m.matches("We are building onchain credit markets.")
    assert m.matches("We are building on-chain credit markets.")
    assert m.matches("We are building onchain-credit markets.")
