import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_ollama_client():
    """Reusable mock for ollama.AsyncClient, capable of streaming and non-streaming."""
    client = AsyncMock()
    
    async def chat_side_effect(model, messages, stream=False, **kwargs):
        if stream:
            async def token_stream():
                yield {"message": {"content": "streamed "}}
                yield {"message": {"content": "response"}}
            return token_stream()
        else:
            return {"message": {"content": "non-streamed response"}}

    client.chat.side_effect = chat_side_effect
    client.list = AsyncMock(return_value={"models": [{"model": "llama3.2:3b"}, {"model": "mistral:7b"}]})
    return client

@pytest.fixture
def mock_search_response():
    """Standard Brave Search API response."""
    return {
        "web": {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "description": "Description 1"
                }
            ]
        }
    }

@pytest.fixture
def mock_cache():
    """Reusable mock for search cache."""
    cache = MagicMock()
    cache.get.return_value = None
    cache.get_by_id.return_value = None
    cache.set.return_value = 42
    cache.get_cached_urls.return_value = {}
    return cache

@pytest.fixture
def mock_http_client():
    """Mock HTTP client for search/scraping."""
    client = AsyncMock()
    client.get = AsyncMock()
    return client

@pytest.fixture
def chat_request():
    """Standard ChatRequest for testing."""
    from models.api_models import ChatRequest
    return ChatRequest(
        model="llama3.2:3b",
        prompt="Test prompt",
        history=[],
        user_memory=""
    )

@pytest.fixture
def chat_context(chat_request, mock_ollama_client):
    """Standard ChatContext for testing."""
    from models.chat_models import ChatContext
    return ChatContext(
        request=chat_request,
        client=mock_ollama_client,
        messages=[{"role": "system", "content": "sys"}],
        system_prompt="sys"
    )

@pytest.fixture
def ollama_client_builder():
    from tests.fixtures.mock_clients import OllamaClientBuilder
    return OllamaClientBuilder()

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {"X-API-Key": "test-key"}

@pytest.fixture
def configured_app(monkeypatch, mock_ollama_client):
    """Pre-configured app with all standard mocks."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth import APIKeyMiddleware
    from routes import chat, chat_stream, models_route
    from config import Config

    monkeypatch.setattr(APIKeyMiddleware, "API_KEY", "test-key")
    monkeypatch.setattr(Config, "OPENWEATHER_API_KEY", "test_owm_key")
    
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)
    app.include_router(chat.router)
    app.include_router(chat_stream.router)
    app.include_router(models_route.router)

    monkeypatch.setattr("routes.chat.ollama.AsyncClient", lambda: mock_ollama_client)
    monkeypatch.setattr("routes.chat_stream.ollama.AsyncClient", lambda: mock_ollama_client)

    with TestClient(app) as client:
        yield client