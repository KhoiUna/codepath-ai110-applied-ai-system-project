"""
Retrieval-Augmented Generation (RAG) mode for the Music Recommender.

Pipeline for a free-text request such as "calm music for late-night studying":

  1. PARSE     Claude turns the request into a taste profile (user_prefs dict),
               constrained by a JSON schema whose genre/mood enums come from
               the catalog itself. Output is re-validated (guardrail).
  2. RETRIEVE  The existing scorer (recommend_songs / score_song) ranks the
               catalog and returns the top-k songs with reasons, plus the
               matching genre/mood notes from data/genre_notes.md.
  3. GENERATE  Claude writes the recommendation using ONLY the retrieved songs
               and notes, returning structured picks.
  4. VERIFY    Every picked title must be in the retrieved set. One corrective
               retry, then fall back to the deterministic ranking.

If no API key is configured, or any API call fails, the pipeline degrades to
keyword parsing + the deterministic recommender instead of crashing.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from src.recommender import recommend_songs

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_QUERY_CHARS = 500
DEFAULT_PROFILE = {
    "favorite_genre": "",
    "favorite_mood": "",
    "target_energy": 0.5,
    "likes_acoustic": False,
}

# Keyword hints used when Claude is unavailable (fallback parser).
ENERGY_HINTS = {
    "calm": 0.3, "study": 0.35, "studying": 0.35, "sleep": 0.2, "relax": 0.35,
    "chill": 0.35, "focus": 0.4, "gym": 0.92, "workout": 0.92, "run": 0.9,
    "running": 0.9, "party": 0.85, "dance": 0.85, "upbeat": 0.8, "hype": 0.9,
}
ACOUSTIC_HINTS = ("acoustic", "unplugged", "organic",
                  "folk", "piano", "calm", "study")


class QueryError(ValueError):
    """Raised when a user query fails input validation."""


@dataclass
class RagResult:
    query: str
    profile: Dict
    retrieved: List[Tuple[Dict, float, str]]
    notes: Dict[str, str]
    summary: str
    picks: List[Dict] = field(default_factory=list)  # [{"title", "reason"}]
    used_llm: bool = False
    grounded: bool = True
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Guardrails / validation
# --------------------------------------------------------------------------

def validate_query(query: Optional[str]) -> str:
    """Reject empty or oversized queries before anything else runs."""
    q = (query or "").strip()
    if not q:
        raise QueryError(
            "Query is empty. Describe what you want to listen to.")
    if len(q) > MAX_QUERY_CHARS:
        raise QueryError(
            f"Query is too long ({len(q)} chars; max {MAX_QUERY_CHARS}).")
    return q


def catalog_vocab(songs: List[Dict]) -> Tuple[List[str], List[str]]:
    """Sorted distinct genres and moods present in the catalog."""
    genres = sorted({s["genre"].strip().lower() for s in songs})
    moods = sorted({s["mood"].strip().lower() for s in songs})
    return genres, moods


def sanitize_profile(raw: Dict, genres: List[str], moods: List[str]) -> Dict:
    """Coerce any parsed profile into a safe user_prefs dict for score_song()."""
    profile = dict(DEFAULT_PROFILE)
    raw = raw if isinstance(raw, dict) else {}

    genre = str(raw.get("favorite_genre", "")).strip().lower()
    profile["favorite_genre"] = genre if genre in genres else ""
    mood = str(raw.get("favorite_mood", "")).strip().lower()
    profile["favorite_mood"] = mood if mood in moods else ""

    try:
        energy = float(
            raw.get("target_energy", DEFAULT_PROFILE["target_energy"]))
        if energy != energy:  # NaN
            raise ValueError
        profile["target_energy"] = min(max(energy, 0.0), 1.0)
    except (TypeError, ValueError):
        logger.warning("Invalid target_energy %r; using default",
                       raw.get("target_energy"))

    acoustic = raw.get("likes_acoustic", False)
    profile["likes_acoustic"] = acoustic if isinstance(
        acoustic, bool) else False
    return profile


def keyword_parse(query: str, genres: List[str], moods: List[str]) -> Dict:
    """Offline fallback: map words in the query onto catalog labels."""
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))
    # Prefer the longest genre that appears (so "indie pop" beats "pop").
    genre = next((g for g in sorted(genres, key=len, reverse=True)
                  if re.search(rf"\b{re.escape(g)}\b", q)), "")
    mood = next((m for m in moods if m in words), "")
    energies = [v for k, v in ENERGY_HINTS.items() if k in words]
    raw = {
        "favorite_genre": genre,
        "favorite_mood": mood,
        "target_energy": sum(energies) / len(energies) if energies else 0.5,
        "likes_acoustic": any(h in words for h in ACOUSTIC_HINTS),
    }
    return sanitize_profile(raw, genres, moods)


def check_grounding(picks: List[Dict], retrieved_titles: List[str]) -> List[str]:
    """Return picked titles that are NOT in the retrieved set (empty == grounded)."""
    allowed = {t.strip().lower() for t in retrieved_titles}
    return [p.get("title", "") for p in picks
            if str(p.get("title", "")).strip().lower() not in allowed]


# --------------------------------------------------------------------------
# Knowledge base retrieval
# --------------------------------------------------------------------------

def load_notes(path: str) -> Dict[str, str]:
    """Parse '## label' sections of the knowledge base into {label: text}."""
    notes: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            current = None
            for line in f:
                if line.startswith("## "):
                    current = line[3:].strip().lower()
                    notes[current] = ""
                elif current and line.strip():
                    notes[current] = (notes[current] + " " +
                                      line.strip()).strip()
    except OSError as e:
        logger.warning(
            "Could not read knowledge base %s (%s); continuing without notes", path, e)
    logger.info("Loaded %d knowledge-base notes from %s", len(notes), path)
    return notes


def retrieve_notes(profile: Dict, retrieved: List[Tuple[Dict, float, str]],
                   notes: Dict[str, str]) -> Dict[str, str]:
    """Notes for the profile's labels plus every label among retrieved songs."""
    labels = [profile["favorite_genre"], profile["favorite_mood"]]
    for song, _, _ in retrieved:
        labels += [song["genre"].lower(), song["mood"].lower()]
    out: Dict[str, str] = {}
    for label in labels:
        if label and label in notes and label not in out:
            out[label] = notes[label]
    return out


# --------------------------------------------------------------------------
# Claude calls
# --------------------------------------------------------------------------

PARSE_SYSTEM = (
    "You convert a listener's free-text music request into a taste profile for a "
    "content-based recommender. Choose favorite_genre and favorite_mood only from the "
    "allowed values; use an empty string if nothing fits. target_energy is 0.0 (very calm) "
    "to 1.0 (very intense). likes_acoustic is true if they want organic/acoustic sound."
)

GENERATE_SYSTEM = (
    "You are a music recommendation assistant. Recommend songs ONLY from the "
    "'Retrieved songs' list in the user message — never invent or add other songs. "
    "Use the scorer's reasons and the genre/mood notes to explain each pick in one "
    "sentence tied to the listener's request. Pick between 1 and 5 songs, best first, "
    "and write a short 1–2 sentence summary."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["title", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "picks"],
    "additionalProperties": False,
}


def make_client():
    """Return an Anthropic client, or None if no credentials are configured."""
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        return None
    import anthropic  # imported lazily so offline mode works without the SDK
    return anthropic.Anthropic(timeout=60.0, max_retries=2)


class LLMError(RuntimeError):
    """Any failure talking to Claude (network, API, refusal, bad JSON)."""


def _call_json(client, model: str, system: str, user: str, schema: Dict) -> Dict:
    """One Claude call constrained to a JSON schema; returns the parsed object."""
    import anthropic

    start = time.perf_counter()
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": schema}},
            # Server-side refusal fallback: if the model declines, the API reruns
            # the request on a suitable fallback model within the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Network error: {e}") from e
    finally:
        logger.info("Claude call (%s) took %.2fs",
                    model, time.perf_counter() - start)

    if response.stop_reason == "refusal":
        raise LLMError("Claude declined the request")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude response was truncated")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse JSON from Claude: {e}") from e


def llm_parse_profile(client, model: str, query: str,
                      genres: List[str], moods: List[str]) -> Dict:
    schema = {
        "type": "object",
        "properties": {
            "favorite_genre": {"type": "string", "enum": genres + [""]},
            "favorite_mood": {"type": "string", "enum": moods + [""]},
            "target_energy": {"type": "number"},
            "likes_acoustic": {"type": "boolean"},
        },
        "required": ["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"],
        "additionalProperties": False,
    }
    user = (f"Allowed genres: {', '.join(genres)}\nAllowed moods: {', '.join(moods)}\n\n"
            f"Request: {query}")
    return _call_json(client, model, PARSE_SYSTEM, user, schema)


def build_context(query: str, retrieved: List[Tuple[Dict, float, str]],
                  notes: Dict[str, str]) -> str:
    lines = [f"Listener request: {query}", "",
             "Retrieved songs (ranked by the scorer):"]
    for i, (song, score, reasons) in enumerate(retrieved, 1):
        lines.append(
            f"{i}. \"{song['title']}\" by {song['artist']} — genre {song['genre']}, "
            f"mood {song['mood']}, energy {song['energy']:.2f}, "
            f"acousticness {song['acousticness']:.2f}, score {score:.2f}. "
            f"Scorer reasons: {reasons}")
    if notes:
        lines += ["", "Genre/mood notes:"]
        lines += [f"- {label}: {text}" for label, text in notes.items()]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

class MusicRAG:
    def __init__(self, songs: List[Dict], notes: Dict[str, str],
                 client=None, model: Optional[str] = None):
        self.songs = songs
        self.notes = notes
        self.client = client
        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.genres, self.moods = catalog_vocab(songs)

    def ask(self, query: str, k: int = 5) -> RagResult:
        query = validate_query(query)
        logger.info("RAG query: %r", query)
        warnings: List[str] = []

        # 1. PARSE
        profile = None
        if self.client is not None:
            try:
                raw = llm_parse_profile(
                    self.client, self.model, query, self.genres, self.moods)
                profile = sanitize_profile(raw, self.genres, self.moods)
            except LLMError as e:
                logger.error("Profile parsing via Claude failed: %s", e)
                warnings.append(
                    f"Claude unavailable for parsing ({e}); used keyword parser.")
        else:
            warnings.append(
                "No ANTHROPIC_API_KEY set; running offline (keyword parser, no AI summary).")
        if profile is None:
            profile = keyword_parse(query, self.genres, self.moods)
        logger.info("Parsed profile: %s", profile)

        # 2. RETRIEVE
        retrieved = recommend_songs(profile, self.songs, k=k)
        notes = retrieve_notes(profile, retrieved, self.notes)
        logger.info("Retrieved song ids: %s; notes: %s",
                    [s["id"] for s, _, _ in retrieved], list(notes))

        result = RagResult(query=query, profile=profile, retrieved=retrieved,
                           notes=notes, summary="", warnings=warnings)
        if self.client is None or not retrieved:
            return self._fallback(result)

        # 3. GENERATE + 4. VERIFY (one corrective retry)
        context = build_context(query, retrieved, notes)
        retrieved_titles = [s["title"] for s, _, _ in retrieved]
        for attempt in (1, 2):
            try:
                answer = _call_json(self.client, self.model,
                                    GENERATE_SYSTEM, context, ANSWER_SCHEMA)
            except LLMError as e:
                logger.error("Generation failed: %s", e)
                warnings.append(f"Claude generation failed ({e}).")
                return self._fallback(result)

            picks = [p for p in answer.get("picks", []) if isinstance(p, dict)]
            bad = check_grounding(picks, retrieved_titles)
            logger.info("Grounding check attempt %d: %d picks, ungrounded=%s",
                        attempt, len(picks), bad)
            if picks and not bad:
                result.summary = str(answer.get("summary", "")).strip()
                result.picks = picks
                result.used_llm = True
                return result
            context += (f"\n\nYour previous answer {'had no picks' if not picks else f'included songs not in the list: {bad}'}. "
                        f"Only choose from: {retrieved_titles}.")

        warnings.append(
            "Claude's answer failed the grounding check twice; showing scorer results.")
        result.grounded = False
        return self._fallback(result)

    @staticmethod
    def _fallback(result: RagResult) -> RagResult:
        """Deterministic answer built from the scorer's own reasons."""
        result.used_llm = False
        result.picks = [{"title": s["title"], "reason": reasons}
                        for s, _, reasons in result.retrieved]
        result.summary = "Top matches from the scoring engine (no AI summary)."
        logger.info("Using deterministic fallback output")
        return result


def print_rag_result(result: RagResult) -> None:
    """Render a RagResult to the terminal."""
    from src.recommender import print_profile, print_recommendations, WIDTH

    for w in result.warnings:
        print(f"  ⚠  {w}")
    print(f"\n  Request: {result.query}")
    print_profile(result.profile)
    if not result.used_llm:
        print_recommendations(result.retrieved)
        return

    by_title = {s["title"].lower(): (s, score)
                for s, score, _ in result.retrieved}
    print(f"\n  🤖 {result.summary}\n")
    for rank, pick in enumerate(result.picks, 1):
        song, score = by_title[pick["title"].strip().lower()]
        print(
            f"  {rank}. {song['title']}  —  {song['artist']}   (score {score:.2f})")
        print(f"     {pick['reason']}")
        print("  " + "-" * (WIDTH - 2))
    if result.notes:
        print("  Sources: songs.csv scorer + notes on " + ", ".join(result.notes))
