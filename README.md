# SHL Conversational Assessment Recommender

A stateless FastAPI service that recommends SHL Individual Test Solutions
through multi-turn dialogue: it clarifies vague hiring requests, recommends
a grounded shortlist, refines that shortlist as constraints change, and
compares assessments — using only data retrieved from a scraped SHL catalog.

## Stack

- **FastAPI + Pydantic** — `/health` and `/chat`, schema enforced exactly as specified.
- **OpenRouter** (`meta-llama/llama-3.3-70b-instruct`) — LLM call, via the OpenAI SDK pointed at OpenRouter's base URL.
- **Custom BM25-style retriever** (`retrieval.py`) — stdlib only, no vector DB. The catalog is a few hundred short text records, so exact term/product-name matching outperforms embedding similarity here and keeps cold starts fast on free hosting.
- **catalog_clean.json** — 377 cleaned Individual Test Solution records scraped from the SHL product catalog.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — loads the catalog at startup, exposes `/health` and `/chat`, never lets an exception break the response schema |
| `schemas.py` | Pydantic request/response models, matching the spec exactly |
| `retrieval.py` | BM25-style hybrid keyword retrieval + fuzzy name matching over the catalog |
| `agent.py` | System prompt, candidate-context builder, OpenRouter call, and grounding validation |
| `catalog_clean.json` | Cleaned catalog (377 items) |
| `replay_harness.py` | Self-test script that replays the 10 provided traces against a running server |
| `requirements.txt` | Python dependencies |

## Setup (local)

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key_here   # openrouter.ai (free tier available)
uvicorn main:app --reload --port 8000
```

The catalog file `catalog_clean.json` is committed to the repo, so no scraping
step is needed to run the service.

## Test locally

```bash
python replay_harness.py http://localhost:8000
```

This replays each of the 10 provided trace conversations against the live
server, checks schema compliance, the 10-item recommendation cap, and the
8-turn cap, and prints what the agent recommended at each turn for comparison
against the expected shortlist in each trace.

## Deploy (Render, free tier)

1. Push this repo to GitHub (already done).
2. In Render: **New → Web Service**, connect this repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `OPENROUTER_API_KEY` in the Render dashboard.
6. Deploy. Free-tier services cold-start on the first request after
   inactivity — the assignment spec allows up to 2 minutes for the first
   `/health` call, so this is expected. Hit `/health` once before
   submitting if the service has been idle.

A `render.yaml` is included for one-click Blueprint deploys (New → Blueprint → select this repo).

## Test against the deployed URL

```bash
python replay_harness.py https://your-app.onrender.com
```

## Design notes

- **Catalog scope.** The scraped feed has no explicit field separating
  Individual Test Solutions from Pre-packaged Job Solutions, so scoping was
  done by filtering against the "Individual Test Solutions" tab on the live
  catalog page during scraping. 377 items is consistent with SHL's publicly
  listed Individual Test Solutions count.
- **Grounding.** Every recommendation returned by the LLM is validated
  against the real catalog before being sent to the user (`retrieval.py:get_by_name`,
  exact match → substring match → Jaccard token-overlap fuzzy match with a
  0.6 confidence threshold). Anything that doesn't resolve to a real catalog
  item is dropped rather than shown, so hallucinated names/URLs can't reach
  the response.
- **Turn-cap safety.** The agent forces `end_of_conversation=true` once the
  conversation reaches turn 7 (of the evaluator's 8-turn cap) if it already
  has a shortlist, so a long-running clarify/refine loop can't run past the
  cap without ever recommending anything.

See `approach.docx` (or the PDF equivalent) for the full write-up on design
trade-offs, prompt design, and evaluation approach.
