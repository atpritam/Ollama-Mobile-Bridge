import pytest
import httpx
import json
from unittest.mock import patch, AsyncMock

from utils.token_manager import TokenManager
from tests.helpers import assert_sse_event
from tests.fixtures.responses import MOCK_BRAVE_SEARCH_API_RESPONSE, MOCK_WEBPAGE_CONTENT


@pytest.fixture(autouse=True)
def set_api_keys_for_recall_flow(monkeypatch):
    """Set necessary API keys for recall integration tests."""
    monkeypatch.setattr("services.search.Config.BRAVE_SEARCH_API_KEY", "dummy-key")
    monkeypatch.setattr("auth.APIKeyMiddleware.API_KEY", "test-key")
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: 8000)

@pytest.fixture
def mock_external_search_and_scrape(monkeypatch):
    """Mocks external HTTP calls for search and scraping."""
    class MockClient:
        async def get(self, url, **kwargs):
            if "search.brave.com" in url:
                return httpx.Response(200, json=MOCK_BRAVE_SEARCH_API_RESPONSE)
            elif "space.com" in url:
                return httpx.Response(200, text=MOCK_WEBPAGE_CONTENT)
            return httpx.Response(404)

    monkeypatch.setattr("utils.http_client.HTTPClientManager.get_search_client", MockClient)
    monkeypatch.setattr("utils.http_client.HTTPClientManager.get_general_client", MockClient)

@pytest.mark.anyio
async def test_initial_search_stores_results_in_cache_and_returns_id(configured_app, auth_headers, ollama_client_builder, mock_external_search_and_scrape):
    """
    Given an initial search, it should store the results in cache and return a search_id in the response.
    """
    ollama_client = (
        ollama_client_builder
        .set_response(1, "GOOGLE: SpaceX launch records")
        .set_response(2, "Based on the search, SpaceX had a historic achievement.")
        .build()
    )
    with patch("routes.chat.ollama.AsyncClient", return_value=ollama_client):
        search_request_payload = {
            "model": "mock-recall-model",
            "prompt": "What were the latest launch records for SpaceX?",
        }
        response = configured_app.post("/chat", json=search_request_payload, headers=auth_headers)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["search_performed"] is True
        assert "search_id" in response_data
        assert response_data["response"] == "Based on the search, SpaceX had a historic achievement."

@pytest.mark.anyio
async def test_recall_flow_uses_cached_results_and_bypasses_new_search(configured_app, auth_headers, ollama_client_builder, mock_external_search_and_scrape, mock_cache):
    """
    Given a recall command, it should retrieve results from cache and bypass performing a new search.
    """
    # First, simulate an initial search to populate the cache
    search_id_to_recall = 999
    cached_content = f"--- Content from: SpaceX Record ---\nSource: https://www.space.com\n{MOCK_WEBPAGE_CONTENT}"
    mock_cache.get_by_id.return_value = (
        cached_content, 
        "https://www.space.com", 
        {"search_type": "GOOGLE", "query": "SpaceX launch records", "search_id": search_id_to_recall}
    )

    # Configure Ollama client for recall flow
    ollama_client = (
        ollama_client_builder
        .set_response(1, f"RECALL: {search_id_to_recall}", stream=True) # LLM returns recall command (streaming, detected in first-line buffer)
        .set_response(2, "Regarding the SpaceX record, it was a historic achievement.", stream=True) # LLM synthesizes response with cached results
        .build()
    )

    from utils.cache import get_search_cache
    cache = get_search_cache()
    cache._search_counter = search_id_to_recall - 1
    returned_id = cache.set(
        search_type="GOOGLE",
        query="SpaceX launch records",
        scraped_contents={"https://www.space.com": MOCK_WEBPAGE_CONTENT},
        summaries=""
    )
    assert returned_id == search_id_to_recall, f"Cache ID mismatch: expected {search_id_to_recall}, got {returned_id}"
    
    with patch("routes.chat_stream.ollama.AsyncClient", return_value=ollama_client), \
         patch("services.search.SearchService.perform_search", new_callable=AsyncMock) as mock_perform_search:
        
        recall_request_payload = {
            "model": "mock-recall-model",
            "prompt": "Tell me more about that record.",
            "history": [
                {"role": "user", "content": "What were the latest launch records for SpaceX?"},
                {"role": "assistant", "content": "...", "search_id": search_id_to_recall}
            ],
            "stream": True,
        }
        
        with configured_app.stream("POST", "/chat/stream", json=recall_request_payload, headers=auth_headers) as response:
            assert response.status_code == 200
            full_body = "".join(chunk for chunk in response.iter_text())
        
        print(f"\n=== FULL RESPONSE BODY ===\n{full_body}\n========================\n")

        # Verify that perform_search was NOT called (recall should use cached results)
        mock_perform_search.assert_not_called()
        
        # Verify cache was used
        retrieved = cache.get_by_id(search_id_to_recall)
        assert retrieved is not None, "Cache should contain the recall result"

        # Verify emitted events and final response
        assert_sse_event(full_body, "status", stage="recalling", message="Let me look at it...")
        # With buffering, response may be in one chunk or multiple tokens
        # Just verify the expected content is present in token events
        from tests.helpers import assert_token_content_contains
        assert_token_content_contains(full_body, "Regarding")
        assert_token_content_contains(full_body, "SpaceX")
        assert_token_content_contains(full_body, "historic")
        assert_token_content_contains(full_body, "achievement")
        
        final_metadata = json.loads(full_body.split('event: done\ndata: ')[-1].strip())
        assert final_metadata["full_response"] == "Regarding the SpaceX record, it was a historic achievement."
        assert final_metadata["search_performed"] is True
        assert final_metadata["search_id"] == search_id_to_recall
        assert final_metadata["source"] == "https://www.space.com"
