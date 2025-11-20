"""
Pydantic data models for API requests and responses.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str
    search_id: Optional[int] = None


class ChatRequest(BaseModel):
    """Chat request model with conversation history."""
    model: str
    prompt: str
    history: Optional[List[Message]] = None
    system_prompt: Optional[str] = None
    user_memory: Optional[str] = Field(None, max_length=200, description="User preferences and context to remember")