import pytest
import httpx
from unittest.mock import AsyncMock, patch, Mock

from services.search import SearchService
from utils.constants import SearchType
from config import Config

@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    """Set a dummy API key for all tests in this module."""
    monkeypatch.setattr(Config, "BRAVE_SEARCH_API_KEY", "dummy-key")

@pytest.fixture
def configured_search_service(chat_context, mock_cache, monkeypatch):
    """Pre-configured SearchService with standard mocks."""
    monkeypatch.setattr(Config, "get_max_html_text_length", lambda model: 4000)
    monkeypatch.setattr(Config, "get_scrape_count", lambda model, search_type: 1)
    monkeypatch.setattr(Config, "MIN_SUMMARY_CHARS", 100)
    
    service = SearchService(chat_context)
    
    with patch("services.search.get_search_cache", lambda: mock_cache):
        yield service, mock_cache

@pytest.mark.anyio
async def test_perform_search_when_cache_hits_returns_cached_result(configured_search_service, mock_http_client):
    """Given a cached query, when perform_search is called, it should return the cached result without making an API call."""
    service, mock_cache = configured_search_service
    mock_cache.get.return_value = ("cached body", "https://cached", {"search_id": 9})

    with patch("services.search.HTTPClientManager.get_search_client", return_value=mock_http_client):
        result, source, search_id = await service.perform_search(SearchType.GOOGLE, "latest news")
        
        assert result == "cached body"
        assert source == "https://cached"
        assert search_id == 9
        mock_http_client.get.assert_not_called()

@pytest.mark.anyio
async def test_perform_search_when_cache_misses_calls_api(configured_search_service, mock_http_client, mock_search_response):
    """Given an uncached query, when perform_search is called, it should call the search API."""
    service, mock_cache = configured_search_service
    mock_cache.get.return_value = None
    
    mock_response = AsyncMock(status_code=200)
    mock_response.json = Mock(return_value=mock_search_response)
    mock_http_client.get.return_value = mock_response

    with patch("services.search.HTTPClientManager.get_search_client", return_value=mock_http_client), \
         patch.object(service, "_process_search_results", new_callable=AsyncMock) as mock_process:
        
        mock_process.return_value = ("processed", "https://source", 99)
        
        result, source, search_id = await service.perform_search(SearchType.GOOGLE, "mars news")

        assert result == "processed"
        assert source == "https://source"
        assert search_id == 99
        mock_http_client.get.assert_called_once()
        mock_process.assert_called_once()

@pytest.mark.parametrize("error_type, status_code, exception, expected_message", [
    ("invalid_key", 401, None, "Invalid API key"),
    ("rate_limit", 429, None, "API rate limit exceeded"),
    ("server_error", 500, None, "Search API error"),
    ("timeout", None, httpx.TimeoutException("timeout"), "Search timed out"),
])
@pytest.mark.anyio
async def test_perform_search_handles_api_errors(
    configured_search_service, mock_http_client, error_type, status_code, exception, expected_message
):
    """Given an API error, when perform_search is called, it should return a user-friendly error message."""
    service, mock_cache = configured_search_service
    mock_cache.get.return_value = None

    if exception:
        mock_http_client.get.side_effect = exception
    else:
        mock_response = AsyncMock(status_code=status_code)
        mock_response.json = Mock(return_value={})
        mock_http_client.get.return_value = mock_response

    with patch("services.search.HTTPClientManager.get_search_client", return_value=mock_http_client):
        result, source, search_id = await service.perform_search(SearchType.GOOGLE, "test query")
        
        assert expected_message in result
        assert source is None
        assert search_id is None

@pytest.mark.anyio
async def test_process_results_handles_infobox_and_web_results(configured_search_service, mock_search_response):
    """Given API data with infobox and web results, when _process_search_results is called, it should combine them."""
    service, mock_cache = configured_search_service
    
    api_data = mock_search_response
    api_data["infobox"] = {"description": "Quick answer", "title": "Info"}

    with patch.object(service, "_scrape_page", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = "Scraped Content"
        results, source, search_id = await service._process_search_results(api_data, SearchType.GOOGLE, "test query")

        assert "Quick answer" in results
        assert "Scraped Content" in results
        assert "Additional Search Results:" in results
        mock_scrape.assert_called_once()
        mock_cache.set.assert_called_once()

@pytest.mark.anyio
async def test_process_results_handles_no_results(configured_search_service):
    """Given empty API data, when _process_search_results is called, it should return 'No results found.'."""
    service, mock_cache = configured_search_service
    
    with patch.object(service, "_scrape_page", new_callable=AsyncMock) as mock_scrape:
        results, source, search_id = await service._process_search_results({"web": {"results": []}}, SearchType.GOOGLE, "query")
        
        assert results == "No results found."
        assert source is None
        assert search_id is None
        mock_scrape.assert_not_called()
        mock_cache.set.assert_not_called()
