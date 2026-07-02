"""
Pydantic schemas for the SHL Conversational Assessment Recommender.
These MUST match the spec exactly - schema deviation breaks the automated evaluator.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # e.g. "K", "P", "A", "K,S" etc - the catalog 'keys' shorthand


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
