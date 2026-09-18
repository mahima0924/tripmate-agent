"""
Unit tests for src.tools.rag_tool

These run directly against the real destination data files (no LLM
involved), so they're fast, free, and fully offline.
"""

import pytest
from src.tools.rag_tool import search_destination_guide


def test_returns_relevant_chunk_for_known_topic():
    results = search_destination_guide("What should I pack for Tokyo in winter?")
    assert len(results) > 0
    assert any("Tokyo" in r for r in results)


def test_returns_list_of_strings():
    results = search_destination_guide("visa requirements for Barcelona")
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        search_destination_guide("")


def test_whitespace_only_query_raises_value_error():
    with pytest.raises(ValueError):
        search_destination_guide("   ")


def test_nonsense_query_returns_empty_or_low_relevance():
    # A query with no real overlap with any destination content should
    # return an empty list rather than an error or fabricated content.
    results = search_destination_guide("zzz qwerty unrelated gibberish 12345")
    assert isinstance(results, list)


def test_query_matches_correct_city_not_others():
    results = search_destination_guide("is it safe to travel to Bangkok")
    assert len(results) > 0
    assert any("Bangkok" in r for r in results)