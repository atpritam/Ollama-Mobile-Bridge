"""
Pydantic data models for API requests and responses.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Chat request model with conversation history."""
    model: str
    prompt: str
    history: Optional[List[Message]] = Field(None, max_length=30)
    system_prompt: Optional[str] = None