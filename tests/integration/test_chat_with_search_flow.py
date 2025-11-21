import pytest
import httpx
import json
from unittest.mock import patch, AsyncMock

from services.search import SearchService
from utils.html_parser import HTMLParser
from utils.token_manager import TokenManager
from tests.helpers import assert_sse_event
from tests.fixtures.responses import MOCK_BRAVE_SEARCH_API_RESPONSE, MOCK_WEBPAGE_CONTENT


@pytest.fixture(autouse=True)
def set_api_keys_for_search_flow(monkeypatch):
    """Set necessary API keys for search integration tests."""
    monkeypatch.setattr("services.search.Config.BRAVE_SEARCH_API_KEY", "dummy-key")
    monkeypatch.setattr("auth.APIKeyMiddleware.API_KEY", "test-key")
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: 8000)

@pytest.fixture
def mock_external_search_api(monkeypatch):
    """Mocks the HTTP client for Brave Search API calls."""
    class MockSearchClient:
        async def get(self, url, headers, params):
            assert "search.brave.com" in url
            assert headers["X-Subscription-Token"] == "dummy-key"
            if params["q"] == "latest news on SpaceX launches":
                return httpx.Response(200, json=MOCK_BRAVE_SEARCH_API_RESPONSE)
            return httpx.Response(404)
    
    monkeypatch.setattr("utils.http_client.HTTPClientManager.get_search_client", MockSearchClient)

@pytest.fixture
def mock_webpage_scraper(monkeypatch):
    """Mocks the webpage scraping function."""
    async def mock_scrape_page(self, client, url: str, title: str, search_type: str, max_length: int):
        assert url == "https://www.space.com/spacex-falcon-9-launch-record-2025"
        parsed_content = HTMLParser.extract_text(MOCK_WEBPAGE_CONTENT, max_length, url)
        result = f"=== Content from: {title} ===\\nSource: {url}\\n{parsed_content}"
        return result
    
    monkeypatch.setattr(SearchService, "_scrape_page", mock_scrape_page)

@pytest.mark.anyio
async def test_chat_stream_emits_search_status_events(configured_app, auth_headers, ollama_client_builder, mock_external_search_api, mock_webpage_scraper):
    """
    Given a prompt triggering a search, when streaming chat, it should emit correct search status events.
    """
    ollama_client = (
        ollama_client_builder
        .set_response(1, "GOOGLE: latest news on SpaceX launches", stream=False) # Search query extraction
        .set_response(2, "Based on the search, SpaceX had a historic achievement with its 100th launch.", stream=True) # Final streamed response
        .build()
    )
    with patch("routes.chat.ollama.AsyncClient", return_value=ollama_client):
        request_payload = {
            "model": "mock-small-model",
            "prompt": "What is the latest news on SpaceX launches?",
            "stream": True,
        }
        with configured_app.stream("POST", "/chat/stream", json=request_payload, headers=auth_headers) as response:
            assert response.status_code == 200
            full_body = "".join(chunk for chunk in response.iter_text())

        assert_sse_event(full_body, "status", stage="thinking")
        assert_sse_event(full_body, "status", stage="searching", message="Searching google for: latest news on SpaceX launches")
        assert_sse_event(full_body, "status", stage="reading_content", message="Reading content from www.space.com")
        assert_sse_event(full_body, "token", content="Based ")
        assert_sse_event(full_body, "token", content="on ")
        assert_sse_event(full_body, "token", content="the ")
        assert_sse_event(full_body, "token", content="search, ")
        assert_sse_event(full_body, "token", content="SpaceX ")
        assert_sse_event(full_body, "token", content="had ")
        assert_sse_event(full_body, "token", content="a ")
        assert_sse_event(full_body, "token", content="historic ")
        assert_sse_event(full_body, "token", content="achievement ")
        assert_sse_event(full_body, "token", content="with ")
        assert_sse_event(full_body, "token", content="its ")
        assert_sse_event(full_body, "token", content="100th ")
        assert_sse_event(full_body, "token", content="launch. ")
        assert_sse_event(full_body, "done")
@pytest.mark.anyio
async def test_chat_stream_injects_search_results_into_llm_context(configured_app, auth_headers, ollama_client_builder, mock_external_search_api, mock_webpage_scraper):
    """
    Given a prompt triggering a search, it should inject search results into the LLM's system prompt.
    """
    ollama_client = (
        ollama_client_builder
        .set_response(1, "GOOGLE: latest news on SpaceX launches", stream=False) # Search query extraction
        .set_response(2, "Based on the search, SpaceX had a historic achievement with its 100th launch.", stream=True) # Final streamed response
        .build()
    )
    with patch("routes.chat.ollama.AsyncClient", return_value=ollama_client):
        request_payload = {
            "model": "mock-small-model",
            "prompt": "What is the latest news on SpaceX launches?",
            "stream": True,
        }
        with configured_app.stream("POST", "/chat/stream", json=request_payload, headers=auth_headers) as response:
            assert response.status_code == 200
            list(response.iter_text()) # Consume the stream to ensure all calls are made
        
        # Verify the second call to ollama.chat (the streaming call) had the search results in its system prompt
        second_call_messages = ollama_client.chat.call_args_list[1].kwargs["messages"]
        system_prompt = next((msg["content"] for msg in second_call_messages if msg["role"] == "system"), "")
        assert "SpaceX Smashes Launch Record in 2025" in system_prompt
        assert "Source: https://www.space.com/spacex-falcon-9-launch-record-2025" in system_prompt

@pytest.mark.anyio
async def test_chat_stream_full_flow_with_search_returns_correct_response_and_metadata(configured_app, auth_headers, ollama_client_builder, mock_external_search_api, mock_webpage_scraper):
    """
    End-to-end test for search-integrated chat flow, verifying final response and metadata.
    """
    ollama_client = (
        ollama_client_builder
        .set_response(1, "GOOGLE: latest news on SpaceX launches", stream=False) # Search query extraction
        .set_response(2, "Based on the search, SpaceX had a historic achievement with its 100th launch.", stream=True) # Final streamed response
        .build()
    )
    with patch("routes.chat.ollama.AsyncClient", return_value=ollama_client):
        request_payload = {
            "model": "mock-small-model",
            "prompt": "What is the latest news on SpaceX launches?",
            "stream": True,
        }
        with configured_app.stream("POST", "/chat/stream", json=request_payload, headers=auth_headers) as response:
            assert response.status_code == 200
            full_body = "".join(chunk for chunk in response.iter_text())
        
        done_event_prefix = 'event: done\ndata: '
        done_event_start = full_body.rfind(done_event_prefix)
        done_event_json_str = full_body[done_event_start + len(done_event_prefix):].strip()
        final_metadata = json.loads(done_event_json_str)

        assert final_metadata["full_response"] == "Based on the search, SpaceX had a historic achievement with its 100th launch."
        assert final_metadata["search_performed"] is True
        assert final_metadata["search_query"] == "latest news on SpaceX launches"
        assert final_metadata["source"] == "https://www.space.com/spacex-falcon-9-launch-record-2025"
        assert "search_id" in final_metadata
