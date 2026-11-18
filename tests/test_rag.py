"""Tests for the RAG pipeline. Claude is mocked — no network or API key needed."""

import json
from types import SimpleNamespace

import pytest

from src.recommender import load_songs
from src.rag import (
    MusicRAG,
    QueryError,
    catalog_vocab,
    check_grounding,
    keyword_parse,
    load_notes,
    sanitize_profile,
    validate_query,
)

SONGS = load_songs("data/songs.csv")
NOTES = load_notes("data/genre_notes.md")
GENRES, MOODS = catalog_vocab(SONGS)


class FakeClient:
    """Mimics client.beta.messages.create, returning queued JSON payloads."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        )


STUDY_PROFILE = {"favorite_genre": "lofi", "favorite_mood": "chill",
                 "target_energy": 0.35, "likes_acoustic": True}


# --- guardrails ------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", None, "x" * 501])
def test_validate_query_rejects_bad_input(bad):
    with pytest.raises(QueryError):
        validate_query(bad)


def test_sanitize_profile_clamps_and_drops_unknown_labels():
    p = sanitize_profile({"favorite_genre": "Polka", "favorite_mood": "CHILL",
                          "target_energy": 7, "likes_acoustic": "yes"}, GENRES, MOODS)
    assert p == {"favorite_genre": "", "favorite_mood": "chill",
                 "target_energy": 1.0, "likes_acoustic": False}


def test_sanitize_profile_handles_garbage():
    p = sanitize_profile({"target_energy": "loud"}, GENRES, MOODS)
    assert p["target_energy"] == 0.5


def test_check_grounding_flags_songs_outside_retrieved_set():
    picks = [{"title": "Library Rain", "reason": ""},
             {"title": "Bohemian Rhapsody", "reason": ""}]
    assert check_grounding(picks, ["Library Rain", "Midnight Coding"]) == [
        "Bohemian Rhapsody"]


def test_keyword_parse_prefers_longest_genre():
    p = keyword_parse("some happy indie pop for a party", GENRES, MOODS)
    assert p["favorite_genre"] == "indie pop"
    assert p["favorite_mood"] == "happy"
    assert p["target_energy"] > 0.7


# --- retrieval ---------------------------------------------------------------

def test_knowledge_base_loaded_for_every_catalog_label():
    assert set(GENRES) | set(MOODS) <= set(NOTES)


# --- full pipeline -----------------------------------------------------------

def test_offline_fallback_without_client():
    result = MusicRAG(SONGS, NOTES, client=None).ask(
        "calm acoustic music for studying", k=3)
    assert not result.used_llm
    assert len(result.retrieved) == 3
    assert any("ANTHROPIC_API_KEY" in w for w in result.warnings)


def test_llm_answer_uses_retrieved_songs():
    answer = {"summary": "Calm picks.", "picks": [
        {"title": "Library Rain", "reason": "Soft and acoustic."},
        {"title": "Midnight Coding", "reason": "Steady lofi beat."}]}
    client = FakeClient([STUDY_PROFILE, answer])
    result = MusicRAG(SONGS, NOTES, client=client).ask(
        "music for late-night studying", k=5)

    assert result.used_llm and result.grounded
    assert [p["title"] for p in result.picks] == [
        "Library Rain", "Midnight Coding"]
    # Retrieved songs and notes were actually sent to Claude in the generation prompt.
    gen_prompt = client.calls[1]["messages"][0]["content"]
    assert "Library Rain" in gen_prompt and "lofi:" in gen_prompt


def test_hallucinated_song_triggers_retry_then_succeeds():
    bad = {"summary": "x", "picks": [{"title": "Fake Song", "reason": "?"}]}
    good = {"summary": "ok", "picks": [
        {"title": "Library Rain", "reason": "calm"}]}
    client = FakeClient([STUDY_PROFILE, bad, good])
    result = MusicRAG(SONGS, NOTES, client=client).ask("study music")
    assert result.used_llm
    assert len(client.calls) == 3
    assert "Fake Song" in client.calls[2]["messages"][0]["content"]


def test_repeated_hallucination_falls_back_to_scorer():
    bad = {"summary": "x", "picks": [{"title": "Fake Song", "reason": "?"}]}
    client = FakeClient([STUDY_PROFILE, bad, bad])
    result = MusicRAG(SONGS, NOTES, client=client).ask("study music")
    assert not result.used_llm and not result.grounded
    assert all(p["title"] != "Fake Song" for p in result.picks)


def test_api_error_falls_back_without_crashing():
    class Boom(FakeClient):
        def _create(self, **kwargs):
            from src.rag import LLMError
            raise LLMError("network down")

    # LLMError raised from inside create propagates like a wrapped API error.
    result = MusicRAG(SONGS, NOTES, client=Boom([])).ask("gym music")
    assert not result.used_llm
    assert result.retrieved
