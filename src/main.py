"""
Command line runner for the Music Recommender Simulation.

Usage:
  python -m src.main                         # demo: fixed taste profiles
  python -m src.main --ask "calm music for late-night studying"   # RAG mode
  python -m src.main --ask "..." -k 3 -v     # top 3, verbose logging
"""

import argparse
import logging
import os
import sys

from src.recommender import (
    load_songs,
    recommend_songs,
    print_profile,
    print_recommendations,
)
from src.rag import MusicRAG, QueryError, load_notes, make_client, print_rag_result

SONGS_PATH = "data/songs.csv"
NOTES_PATH = "data/genre_notes.md"
LOG_PATH = "logs/app.log"


# Taste profiles. Keys match the UserProfile fields in recommender.py.

# Upbeat listener who wants danceable, polished, high-energy pop.
HIGH_ENERGY_POP = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.85,
    "likes_acoustic": False,
}

# Late-night studier who wants calm, acoustic background music.
CHILL_LOFI = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.40,
    "likes_acoustic": True,
}

# Gym / commute listener who wants loud, driving, electric rock.
DEEP_INTENSE_ROCK = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 0.92,
    "likes_acoustic": False,
}

USER_PROFILES = {
    "High-Energy Pop": HIGH_ENERGY_POP,
    "Chill Lofi": CHILL_LOFI,
    "Deep Intense Rock": DEEP_INTENSE_ROCK,
}


def setup_logging(verbose: bool) -> None:
    """Full INFO log to logs/app.log; console shows warnings (or INFO with -v)."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [file_handler, console]
    # Keep third-party HTTP chatter out of the log.
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_profiles(songs: list, k: int) -> None:
    for name, user_prefs in USER_PROFILES.items():
        recommendations = recommend_songs(user_prefs, songs, k=k)
        print()
        print(f"=== {name} ===")
        print_profile(user_prefs)
        print_recommendations(recommendations)


def run_ask(songs: list, query: str, k: int) -> int:
    rag = MusicRAG(songs, load_notes(NOTES_PATH), client=make_client())
    try:
        result = rag.ask(query, k=k)
    except QueryError as e:
        logging.getLogger(__name__).warning("Rejected query: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print_rag_result(result)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Music Recommender (with RAG 'ask' mode)")
    parser.add_argument("--ask", metavar="QUERY",
                        help="describe what you want in plain English (RAG mode)")
    parser.add_argument("-k", type=int, default=5,
                        help="number of songs to retrieve (1-10)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show INFO logs on console")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()  # picks up ANTHROPIC_API_KEY from .env if present
    except ImportError:
        pass

    setup_logging(args.verbose)
    if not 1 <= args.k <= 10:
        parser.error("-k must be between 1 and 10")

    try:
        songs = load_songs(SONGS_PATH)
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).error("Could not load catalog: %s", e)
        print(f"Error: could not load {SONGS_PATH}: {e}", file=sys.stderr)
        return 1

    if args.ask is not None:
        return run_ask(songs, args.ask, args.k)
    run_profiles(songs, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
