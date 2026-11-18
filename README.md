# 🎵 Music Recommender with RAG "Ask the DJ" Mode

## Project Summary

**Original project (Modules 1–3): _Music Recommender Simulation_.** It was a content-based recommender that ranks a 15-song catalog against a fixed taste profile (genre, mood, target energy, acoustic preference). It uses a transparent, weighted scoring rule. Each recommendation came with a score and a "why" list, so the project was a way to explore how recommenders turn data into rankings and where bias creeps in.

**What it does now:** you describe what you want in plain English, such as _"something sad and reflective for a rainy evening"_, and the app answers with grounded recommendations. Claude turns the request into a taste profile, and the original scorer **retrieves** the best-matching songs plus genre/mood notes. Claude then explains its picks **using only that retrieved data**, and a grounding check rejects any song that isn't in the retrieved set.

**Why it matters:** the language model makes the app easy to use, while the deterministic retriever and the verification step keep the answers factual. They stop it from recommending songs that don't exist in the catalog. This is the core pattern behind trustworthy LLM features in production.

---

## How The System Works

Real platforms like Spotify and YouTube combine **collaborative filtering** (what similar users play) with **content-based filtering** (what a song sounds like). They first score many candidate songs, then rank the list for relevance and variety. This simulation has no data from other users, so it is purely **content-based**. It prioritizes **vibe**, rewarding songs whose features are _closest_ to the user's preferences rather than simply higher or lower. Each song gets a score, and the top-k scores become the recommendations.

**`Song` features used**

- `energy`: main signal, scored by closeness to the user's target
- `acousticness`: organic vs. electronic sound
- `mood`: bonus for an exact match (weighted above genre)
- `genre`: smaller bonus for an exact match
- `title`, `artist`: shown in the output and explanation, not scored

**`UserProfile` fields**

- `favorite_genre`
- `favorite_mood`
- `target_energy` (0–1)
- `likes_acoustic` (True/False)

### Data Flow

```
INPUT                      PROCESS (the loop)                         OUTPUT (the ranking)
─────                      ──────────────────                         ────────────────────
user_prefs  ──┐            for each song in songs:                    sort by score (high → low)
              ├──────────►     score, reasons = score_song(prefs, song)  ──►  keep top k
songs.csv ────┘                collect (song, score, reasons)              print title, score, "Because: ..."
(load_songs)
```

1. **Input:** `load_songs()` reads `data/songs.csv` into a list of dicts and converts the numeric columns to floats. `main.py` supplies the user's taste profile.
2. **Process:** `recommend_songs()` loops over **every** song and calls `score_song()` on each one. `score_song()` judges a single song in isolation and returns its score along with a list of reasons, one for each rule that added points.
3. **Output:** The scored songs are sorted from highest to lowest score and cut to the top `k`. Each recommendation is printed with its score and the reasons behind it.

### Algorithm Recipe (finalized)

| Signal           | Points         | Rule                                                                      |
| ---------------- | -------------- | ------------------------------------------------------------------------- |
| Mood match       | **+2.0**       | `song.mood == favorite_mood`                                              |
| Genre match      | **+1.0**       | `song.genre == favorite_genre`                                            |
| Energy closeness | **0.0 to 2.0** | `2.0 × (1 − abs(song.energy − target_energy))`                            |
| Acoustic fit     | **0.0 to 0.5** | `0.5 × acousticness` if `likes_acoustic`, else `0.5 × (1 − acousticness)` |

The maximum possible score is about **5.5**.

**Why these weights?**

- **Mood counts double compared to genre.** In this catalog, genre labels are nearly unique: 14 genres across 15 songs. A large genre bonus ends up picking a single song instead of describing a taste. Mood groups songs by how they feel across genres. For example, "chill" covers both lofi and ambient songs.
- **Energy can be worth as much as mood.** A song whose energy is 0.5 away from the target loses 1 point. That stops a matching label from hiding a big mismatch in feel.
- **Acoustic fit is a small nudge.** It mostly breaks ties between songs that are otherwise similar.

I tested this against the common starting recipe (+2.0 genre, +1.0 mood, energy × 1.0). For a user who wants "pop / chill / energy 0.30", the starting recipe ranked _Gym Hero_ (intense, energy 0.93) second just because it is pop. The final recipe recommended _Spacewalk Thoughts_, _Library Rain_ and _Midnight Coding_ instead, which are actually calm songs.

### Expected Biases

- **Mood over-prioritization:** Mood is the largest single bonus, so moods with several songs dominate the top results. "chill" has 3 songs, while a "nostalgic" or "angry" listener has only one exact match. Great songs with a neighboring mood (for example, "relaxed" vs. "chill") get no credit.
- **Exact string matching:** "indie pop" does not match "pop", and "relaxed" does not match "chill". The system has no idea which labels are related.
- **Catalog imbalance:** Lofi has 3 songs and most genres have 1. Lofi and chill listeners get many good options, while fans of other genres get one match followed by songs that are only close in energy.
- **Energy as the main signal:** Two songs with the same energy but very different sounds can score the same, because tempo, valence and danceability are ignored.
- **Tie-breaking by CSV order:** Python's sort is stable, so songs with equal scores appear in the order they are listed in the file, not by any merit.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

   ```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. (Optional, for AI mode) Add your Anthropic API key:

```bash
cp .env.example .env      # then edit .env and set ANTHROPIC_API_KEY=...
# or: export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, `--ask` still works. It runs offline with a keyword parser and the plain scorer output, and prints a warning.

4. Run the app (always from the project root):

```bash
python -m src.main                                              # demo with 3 fixed taste profiles
python -m src.main --ask "calm acoustic music for late-night studying"   # RAG mode
python -m src.main --ask "upbeat songs for the gym" -k 3 -v     # top 3, show logs on console
```

### Running Tests

```bash
pytest
```

`tests/test_rag.py` mocks Claude, so the tests need no network access and no API key.

---

## AI Feature: Retrieval-Augmented Generation ("Ask" mode)

`--ask` lets you describe what you want in plain English. Claude only writes its answer from songs and notes that the app **retrieved** first (`src/rag.py`):

```
"calm music for studying"
   │
   ▼ 1. PARSE     Claude → taste profile (JSON schema; genre/mood enums = catalog labels)
   │              guardrail: re-validate, clamp energy 0–1, drop unknown labels
   ▼ 2. RETRIEVE  existing scorer (recommend_songs) → top-k songs + reasons
   │              + matching entries from data/genre_notes.md (knowledge base)
   ▼ 3. GENERATE  Claude writes picks + reasons using ONLY the retrieved context
   ▼ 4. VERIFY    every picked title must be in the retrieved set
                  → one corrective retry → else fall back to scorer output
```

**Guardrails and logging**

- Queries that are empty or longer than 500 characters are rejected with exit code 2.
- If the API key is missing, the API returns an error, the network fails, Claude refuses, or the JSON is malformed, the app falls back to the deterministic recommender instead of crashing.
- Requests use the API's server-side refusal fallback (`fallbacks: "default"`).
- Every run writes to `logs/app.log`: the parsed profile, the retrieved song IDs, the notes used, the latency of each Claude call, and the result of each grounding check. Add `-v` to also print these logs to the console.
- The model defaults to `claude-opus-5-5`. You can override it with `ANTHROPIC_MODEL`.

---

## Architecture Overview

The full diagram is in [`diagrams/system_architecture.mmd`](diagrams/system_architecture.mmd) (Mermaid source).

- **Input:** the user's request, the song catalog (`data/songs.csv`), and the knowledge base (`data/genre_notes.md`).
- **Parser (Claude):** maps the request onto catalog labels. A keyword parser is the backup, and `sanitize_profile` is a guardrail on the result.
- **Retriever:** the original `recommend_songs`/`score_song` engine picks the top-k songs with reasons, and `retrieve_notes` adds the matching knowledge-base entries.
- **Generator (Claude):** writes the picks and reasons from the retrieved context only.
- **Evaluator:** `check_grounding` confirms every pick was retrieved. It allows one corrective retry, then falls back to the deterministic scorer.
- **Output:** the terminal answer plus `logs/app.log`.
- **Human and testing checks:** `pytest` (with Claude mocked) checks the guardrails, grounding and fallbacks. A developer reviews the logs and output, and the user judges whether the picks fit.

---

## Sample Interactions

These are real outputs from `python -m src.main --ask "..."` using `claude-opus-5-5`, with the profile header trimmed.

**1. "upbeat songs for the gym"** → parsed as `edm / intense / energy 0.90 / non-acoustic`

```
🤖 High-energy, driving tracks to power your workout, led by intense rock and pop picks
   with EDM, metal, and an upbeat pop closer for variety.
1. Storm Runner — Voltline (score 4.43)
   Its intense mood and 0.91 energy nearly hit the target exactly, and its loud, driving
   rock sound is built for pushing through a hard set.
2. Gym Hero — Max Pulse (score 4.42)
   This hook-driven pop track pairs an intense, high-adrenaline mood with 0.93 energy...
3. Neon Horizon — Pulse Theory (score 3.36)
   An EDM track with big drops and 0.96 energy...
Sources: songs.csv scorer + notes on edm, intense, rock, pop, euphoric, metal, angry, happy
```

**2. "calm acoustic music for late-night studying"** → `lofi / focused / energy 0.25 / acoustic`

```
🤖 For calm, acoustic late-night studying, these picks lean on low-energy lofi and ambient
   tracks that keep you focused without demanding attention, plus a highly acoustic classical option.
1. Focus Flow — LoRoom (score 5.09)
   Its focused mood and lofi genre both match the request, and its steady, unobtrusive feel...
2. Library Rain — Paper Lanterns (score 3.23)
   A chill lofi track with soft, warm textures and high acousticness (0.86)...
5. Winter Sonata — Elena Marsh (score 2.42)
   ...very high acousticness (0.97) fit a calm acoustic session, though its melancholic
   mood is more emotional than focused.
```

**3. "something sad and reflective for a rainy evening"** → `any genre / melancholic / energy 0.25 / acoustic`

```
🤖 Only "Winter Sonata" truly matches the sad, reflective mood you asked for; the other picks
   are calm and low-key rather than sad, so they work as gentle companions for a rainy evening.
1. Winter Sonata — Elena Marsh (score 4.42)
   It's the only melancholic match, a wistful, highly acoustic classical piece at very low energy...
2. Spacewalk Thoughts — Orbit Bloom (score 2.40)
   ...though its mood is chill rather than sad.
```

Example 3 shows the RAG benefit: the AI is honest about the catalog's limits (only one melancholic song) instead of inventing sad songs. When the API was unreachable (for example, with no credits), the same command printed a warning and the deterministic scorer ranking instead of crashing.

---

## Design Decisions

- **Reuse the original scorer as the retriever.** Retrieval stays transparent and testable, and it is the same logic documented in the Algorithm Recipe. The trade-off is that retrieval quality is capped by the simple scoring rule.
- **The LLM only translates and explains; it does not rank.** Constrained JSON output, with genre/mood enums built from the catalog, keeps Claude's profile compatible with `score_song`. A post-generation grounding check catches hallucinated songs.
- **Structured output instead of free text.** Picks come back as `{title, reason}` JSON, so grounding is an exact title check rather than fuzzy text matching.
- **Graceful degradation over hard failure.** No key, an API error, a refusal or bad JSON all fall back to the deterministic recommender. The trade-off is that the user may get a less personalized answer, but always gets one.
- **Small Markdown knowledge base** (`genre_notes.md`). It is easy to read and edit, and keyword retrieval is enough for 23 labels. A larger catalog would need embeddings.
- **Trade-off accepted:** Claude may reorder the retrieved songs. In example 2 it ranked _Spacewalk Thoughts_ (2.40) above _Midnight Coding_ (3.02). The system allows this because the reasons stay grounded, but the scores shown can look out of order.

---

## Testing Summary

- **16 automated tests pass** (`pytest`, about 1 second, no network):
  - input guardrails (empty or over 500-character queries)
  - profile sanitization (unknown labels, out-of-range energy)
  - keyword fallback parsing
  - knowledge-base coverage of every catalog label
  - a full pipeline run with a mocked Claude
  - a hallucinated song triggering a retry
  - repeated hallucination falling back to the scorer
  - API errors not crashing the app
- **Manual runs** with a real key produced grounded answers on the first attempt (the logs show `ungrounded=[]`).
- **What didn't work at first:** a live run failed with `400 credit balance too low`. The fallback path handled it correctly, and the error was logged with its request ID, which confirmed that the guardrails work in practice.
- **What I learned:** mocking the LLM client makes AI code testable and deterministic. The most valuable tests cover failure paths, not the happy path.

---

## Sample Output: Profile Demo Mode (`python -m src.main`)

```
================================================================
  🎵  MUSIC RECOMMENDER
================================================================
  Genre: lofi  |  Mood: chill  |  Energy: 0.40  |  Acoustic: yes
================================================================

  Top 5 recommendations

  1. Library Rain  —  Paper Lanterns
     Score: 5.33 / 5.5  ███████████████████░
     Why:
       • mood match: chill (+2.00)
       • genre match: lofi (+1.00)
       • energy 0.35 vs target 0.40 (+1.90)
       • acoustic fit: acousticness 0.86 (+0.43)
  --------------------------------------------------------------
  2. Midnight Coding  —  LoRoom
     Score: 5.31 / 5.5  ███████████████████░
     Why:
       • mood match: chill (+2.00)
       • genre match: lofi (+1.00)
       • energy 0.42 vs target 0.40 (+1.96)
       • acoustic fit: acousticness 0.71 (+0.35)
  --------------------------------------------------------------
  3. Spacewalk Thoughts  —  Orbit Bloom
     Score: 4.22 / 5.5  ███████████████░░░░░
     Why:
       • mood match: chill (+2.00)
       • energy 0.28 vs target 0.40 (+1.76)
       • acoustic fit: acousticness 0.92 (+0.46)
  --------------------------------------------------------------
  4. Focus Flow  —  LoRoom
     Score: 3.39 / 5.5  ████████████░░░░░░░░
     Why:
       • genre match: lofi (+1.00)
       • energy 0.40 vs target 0.40 (+2.00)
       • acoustic fit: acousticness 0.78 (+0.39)
  --------------------------------------------------------------
  5. Coffee Shop Stories  —  Slow Stereo
     Score: 2.38 / 5.5  █████████░░░░░░░░░░░
     Why:
       • energy 0.37 vs target 0.40 (+1.94)
       • acoustic fit: acousticness 0.89 (+0.45)
  --------------------------------------------------------------
```

Multiple profiles:

```
=== High-Energy Pop ===
================================================================
  🎵  MUSIC RECOMMENDER
================================================================
  Genre: pop  |  Mood: happy  |  Energy: 0.85  |  Acoustic: no
================================================================

  Top 5 recommendations

  1. Sunrise City  —  Neon Echo
     Score: 5.35 / 5.5  ███████████████████░
     Why:
       • mood match: happy (+2.00)
       • genre match: pop (+1.00)
       • energy 0.82 vs target 0.85 (+1.94)
       • non-acoustic fit: acousticness 0.18 (+0.41)
  --------------------------------------------------------------
  2. Rooftop Lights  —  Indigo Parade
     Score: 4.15 / 5.5  ███████████████░░░░░
     Why:
       • mood match: happy (+2.00)
       • energy 0.76 vs target 0.85 (+1.82)
       • non-acoustic fit: acousticness 0.35 (+0.33)
  --------------------------------------------------------------
  3. Gym Hero  —  Max Pulse
     Score: 3.31 / 5.5  ████████████░░░░░░░░
     Why:
       • genre match: pop (+1.00)
       • energy 0.93 vs target 0.85 (+1.84)
       • non-acoustic fit: acousticness 0.05 (+0.47)
  --------------------------------------------------------------
  4. Storm Runner  —  Voltline
     Score: 2.33 / 5.5  ████████░░░░░░░░░░░░
     Why:
       • energy 0.91 vs target 0.85 (+1.88)
       • non-acoustic fit: acousticness 0.10 (+0.45)
  --------------------------------------------------------------
  5. Crown Talk  —  Verse Kingdom
     Score: 2.30 / 5.5  ████████░░░░░░░░░░░░
     Why:
       • energy 0.78 vs target 0.85 (+1.86)
       • non-acoustic fit: acousticness 0.12 (+0.44)
  --------------------------------------------------------------

=== Chill Lofi ===
================================================================
  🎵  MUSIC RECOMMENDER
================================================================
  Genre: lofi  |  Mood: chill  |  Energy: 0.40  |  Acoustic: yes
================================================================

  Top 5 recommendations

  1. Library Rain  —  Paper Lanterns
     Score: 5.33 / 5.5  ███████████████████░
     Why:
       • mood match: chill (+2.00)
       • genre match: lofi (+1.00)
       • energy 0.35 vs target 0.40 (+1.90)
       • acoustic fit: acousticness 0.86 (+0.43)
  --------------------------------------------------------------
  2. Midnight Coding  —  LoRoom
     Score: 5.31 / 5.5  ███████████████████░
     Why:
       • mood match: chill (+2.00)
       • genre match: lofi (+1.00)
       • energy 0.42 vs target 0.40 (+1.96)
       • acoustic fit: acousticness 0.71 (+0.35)
  --------------------------------------------------------------
  3. Spacewalk Thoughts  —  Orbit Bloom
     Score: 4.22 / 5.5  ███████████████░░░░░
     Why:
       • mood match: chill (+2.00)
       • energy 0.28 vs target 0.40 (+1.76)
       • acoustic fit: acousticness 0.92 (+0.46)
  --------------------------------------------------------------
  4. Focus Flow  —  LoRoom
     Score: 3.39 / 5.5  ████████████░░░░░░░░
     Why:
       • genre match: lofi (+1.00)
       • energy 0.40 vs target 0.40 (+2.00)
       • acoustic fit: acousticness 0.78 (+0.39)
  --------------------------------------------------------------
  5. Coffee Shop Stories  —  Slow Stereo
     Score: 2.38 / 5.5  █████████░░░░░░░░░░░
     Why:
       • energy 0.37 vs target 0.40 (+1.94)
       • acoustic fit: acousticness 0.89 (+0.45)
  --------------------------------------------------------------

=== Deep Intense Rock ===
================================================================
  🎵  MUSIC RECOMMENDER
================================================================
  Genre: rock  |  Mood: intense  |  Energy: 0.92  |  Acoustic: no
================================================================

  Top 5 recommendations

  1. Storm Runner  —  Voltline
     Score: 5.43 / 5.5  ████████████████████
     Why:
       • mood match: intense (+2.00)
       • genre match: rock (+1.00)
       • energy 0.91 vs target 0.92 (+1.98)
       • non-acoustic fit: acousticness 0.10 (+0.45)
  --------------------------------------------------------------
  2. Gym Hero  —  Max Pulse
     Score: 4.46 / 5.5  ████████████████░░░░
     Why:
       • mood match: intense (+2.00)
       • energy 0.93 vs target 0.92 (+1.98)
       • non-acoustic fit: acousticness 0.05 (+0.47)
  --------------------------------------------------------------
  3. Neon Horizon  —  Pulse Theory
     Score: 2.41 / 5.5  █████████░░░░░░░░░░░
     Why:
       • energy 0.96 vs target 0.92 (+1.92)
       • non-acoustic fit: acousticness 0.03 (+0.48)
  --------------------------------------------------------------
  4. Iron Verdict  —  Black Anvil
     Score: 2.37 / 5.5  █████████░░░░░░░░░░░
     Why:
       • energy 0.98 vs target 0.92 (+1.88)
       • non-acoustic fit: acousticness 0.02 (+0.49)
  --------------------------------------------------------------
  5. Sunrise City  —  Neon Echo
     Score: 2.21 / 5.5  ████████░░░░░░░░░░░░
     Why:
       • energy 0.82 vs target 0.92 (+1.80)
       • non-acoustic fit: acousticness 0.18 (+0.41)
  --------------------------------------------------------------
```

**Screenshot or video** _(optional)_: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

This project taught me that an LLM is most reliable when it isn't trusted on its own. Pairing it with a deterministic retriever, a schema, and a verification step turned a "creative" model into a dependable feature. Building the fallbacks first also made the system resilient to real failures, such as running out of API credits. Finally, the AI's answers made the catalog's biases visible (for example, only one melancholic song), which is a reminder that a recommender can only be as fair as its data.

The full responsible-AI reflection is in the [**Model Card**](model_card.md): how I collaborated with AI, the suggestions I accepted and rejected, and the system's limitations.
