# SHL Conversational Assessment Recommender

## Setup (local)

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # console.groq.com (free)
uvicorn main:app --reload --port 8000
```

Catalog file `catalog_clean.json` is already built (377 items, cleaned from
the raw scraped feed). If you re-scrape, re-run the cleaning step shown in
the chat history / regenerate with the same field mapping.

## Test locally

```bash
python replay_harness.py http://localhost:8000
```

This replays all 10 of your trace conversations' USER turns against your
live local server, checks schema compliance, the 10-item cap, and the
8-turn cap, and prints out what your agent actually recommended at each
turn so you can eyeball it against the expected shortlist in each trace
.md file.

**Do this BEFORE deploying** — much faster iteration loop locally.

## Deploy (Render free tier)

1. Push this folder to a GitHub repo.
2. New Web Service on Render, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `GROQ_API_KEY` in Render's dashboard.
6. Deploy. First `/health` call after a cold start can take up to ~1 min on
   free tier — the spec explicitly allows up to 2 minutes for this, so it's
   fine, but test it once after a period of inactivity before submitting.

## Test against deployed URL

```bash
python replay_harness.py https://your-app.onrender.com
```

## Known things to verify before submission (flagged during build)

1. **`recommendations: null` vs `[]`** — the trace .md files write `null`
   when not recommending, but the written spec's example schema shows an
   array type. Current implementation always returns `[]` (never `null`).
   If the automated evaluator strictly requires `null`, this needs a
   one-line change in `schemas.py` (`recommendations: Optional[List[...]] = None`)
   and `agent.py` (return `None` instead of `[]` in clarify/refuse states).
   **Test this against the harness output and decide.**

2. **Catalog scope (Individual Test Solutions only)** — the scraped feed
   has no explicit category field distinguishing Individual Test Solutions
   from Pre-packaged Job Solutions. 377 items is consistent with SHL's
   publicly known Individual Test Solutions count, so it's assumed already
   correctly scoped — but cross-check against the live catalog page filter
   toggle before submitting, since getting this wrong fails a hard eval.

3. **Fuzzy name matching threshold** (`retrieval.py` `get_by_name`, 0.6
   Jaccard/coverage blend) — tune this if you see either (a) valid LLM
   recommendations being wrongly stripped as "ungrounded" in logs, or
   (b) wrong catalog items being matched to a close-but-different name.

## Files

- `main.py` — FastAPI app, `/health` and `/chat`
- `schemas.py` — Pydantic request/response models
- `retrieval.py` — BM25-style hybrid retrieval over the catalog
- `agent.py` — system prompt + Groq call + grounding validation
- `catalog_clean.json` — cleaned catalog (377 items)
- `replay_harness.py` — self-test against your 10 traces
- `requirements.txt`
