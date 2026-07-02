"""
SHL Conversational Assessment Recommender — FastAPI service.
"""
import os
import json
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from retrieval import CatalogRetriever
from agent import run_agent_turn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Assessment Recommender")

CATALOG_PATH = os.environ.get("CATALOG_PATH", "catalog_clean.json")

retriever: CatalogRetriever | None = None


@app.on_event("startup")
def load_catalog():
    global retriever
    if not os.path.exists(CATALOG_PATH):
        logger.warning(f"Catalog file {CATALOG_PATH} not found at startup — /chat will fail until it exists.")
        retriever = None
        return
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)
    retriever = CatalogRetriever(catalog)
    logger.info(f"Loaded catalog with {len(catalog)} items.")


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if retriever is None:
        # Fail safe rather than crash — still schema-compliant.
        return ChatResponse(
            reply="The assessment catalog is temporarily unavailable. Please try again shortly.",
            recommendations=[],
            end_of_conversation=False,
        )

    try:
        result = run_agent_turn(request.messages, retriever)
        return result
    except Exception as e:
        logger.exception("Error in /chat")
        # Hard-eval safety net: never 500, never break schema.
        return ChatResponse(
            reply="Sorry, I hit an internal error processing that. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )
