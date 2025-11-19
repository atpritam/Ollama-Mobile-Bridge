"""
Data models for chat processing.
Contains context objects, search results, and flow control structures.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import ollama
from models.api_models import ChatRequest


@dataclass
class ChatContext:
    """
    Context object containing all chat processing state.
    Provides centralized access to request-scoped data, eliminating parameter chaining.
    """
    request: ChatRequest
    client: ollama.AsyncClient
    messages: list
    system_prompt: str
    call_count: int = 0

    @property
    def model_name(self) -> str:
        """Get model name from request."""
        return self.request.model

    @property
    def prompt(self) -> str:
        """Get user prompt from request."""
        return self.request.prompt

    @property
    def user_memory(self) -> str | None:
        """Get user memory from request."""
        return self.request.user_memory

    def next_call_number(self) -> int:
        """Increment and return the next LLM call number."""
        self.call_count += 1
        return self.call_count


@dataclass
class SearchResult:
    """Search result information."""
    performed: bool
    search_type: Optional[str] = None
    search_query: Optional[str] = None
    search_results: Optional[str] = None
    source_url: Optional[str] = None
    search_id: Optional[int] = None


class FlowAction(Enum):
    """Types of actions in the chat flow."""
    SEARCH = "search"
    CALL_LLM = "call_llm"
    STREAM_RESPONSE = "stream_response"
    RETURN_RESPONSE = "return_response"


@dataclass
class FlowStep:
    """Represents a step in the chat processing flow."""
    action: FlowAction
    search_type: Optional[str] = None
    search_query: Optional[str] = None
    messages: Optional[list] = None
    response: Optional[str] = None
    search_result: Optional[SearchResult] = None
    call_number: Optional[int] = None