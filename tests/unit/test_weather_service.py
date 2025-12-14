import pytest
from unittest.mock import AsyncMock, patch, Mock, MagicMock

from services.weather import WeatherService
from tests.fixtures.responses import DEFAULT_WEATHER_RESPONSE
from config import Config


@pytest.fixture(autouse=True)
def mock_weather_cache(monkeypatch):
    """Fixture to provide a fresh mock cache for weather tests."""
    cache = MagicMock()
    cache.get.return_value = None
    cache.get_by_id.return_value = None
    cache.set.return_value = 42
    cache.get_cached_urls.return_value = {}
    monkeypatch.setattr("utils.cache._search_cache", cache)
    return cache

@pytest.fixture
def mock_search_service():
    """Fixture for a mocked SearchService."""
    service = Mock()
    service.perform_search = AsyncMock(return_value=("Scraped weather", "http://example.com/scraped", 123))
    return service

@pytest.fixture(autouse=True)
def set_openweathermap_api_key(monkeypatch):
    """Set a dummy API key for OpenWeatherMap."""
    monkeypatch.setattr(Config, "OPENWEATHER_API_KEY", "test_key")


@pytest.mark.anyio
async def test_get_weather_when_api_call_is_successful(mock_http_client, mock_search_service, monkeypatch, mock_weather_cache):
    """Given a successful API response, get_weather should return formatted weather data."""
    monkeypatch.setattr("utils.http_client.HTTPClientManager.get_search_client", lambda: mock_http_client)
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.json = Mock(return_value=DEFAULT_WEATHER_RESPONSE)
    mock_http_client.get.return_value = response_mock

    weather, source, search_id = await WeatherService.get_weather("Test City", mock_search_service)

    assert "Current Weather in Test City" in weather
    assert "Conditions: Clear (Clear sky)" in weather
    assert "Temperature: 25°C" in weather
    assert source == "https://openweathermap.org/city/12345"
    assert search_id == 42
    mock_weather_cache.set.assert_called_once()
    mock_search_service.perform_search.assert_not_called()


@pytest.mark.anyio
async def test_get_weather_when_city_not_found_by_api(mock_http_client, mock_search_service, monkeypatch, mock_weather_cache):
    """Given an API response indicating city not found, get_weather should return a specific message."""
    mock_weather_cache.get.return_value = None

    monkeypatch.setattr("utils.http_client.HTTPClientManager.get_search_client", lambda: mock_http_client)
    mock_http_client.get.return_value = AsyncMock(status_code=404)

    weather, source, search_id = await WeatherService.get_weather("Unknown City", mock_search_service)

    assert "City not found" in weather
    assert source is None
    assert search_id == 42
    mock_weather_cache.set.assert_called_once()
    mock_search_service.perform_search.assert_not_called()

@pytest.mark.parametrize("api_key, http_status, http_exception, expected_search_query", [
    ("invalid_key", 401, None, "Test City weather today"),
    ("test_key", 500, None, "Test City weather today"),
    ("test_key", None, Exception("API Down"), "Test City weather today"),
    (None, None, None, "Test City weather today"),
])
@pytest.mark.anyio
async def test_get_weather_fallbacks_to_search_on_api_issues(
    mock_http_client, mock_search_service, monkeypatch, api_key, http_status, http_exception, expected_search_query, mock_weather_cache
):
    """Given API issues (invalid key, error, no key), get_weather should fallback to SearchService.perform_search."""

    if api_key is None:
        monkeypatch.setattr(Config, "OPENWEATHER_API_KEY", None)
    else:
        monkeypatch.setattr(Config, "OPENWEATHER_API_KEY", api_key)
    
    if http_exception:
        mock_http_client.get.side_effect = http_exception
    else:
        mock_http_client.get.return_value = AsyncMock(status_code=http_status)

    weather, source, search_id = await WeatherService.get_weather("Test City", mock_search_service)

    assert weather == "Scraped weather"
    assert source == "http://example.com/scraped"
    assert search_id == 123
    mock_search_service.perform_search.assert_awaited_once_with("google", expected_search_query)
    mock_weather_cache.set.assert_not_called() # Fallback implies cache.set for API was not called

@pytest.mark.anyio
async def test_get_weather_when_cache_hits(mock_http_client, mock_search_service, monkeypatch, mock_weather_cache):
    """Given a cached weather result, get_weather should return it without making API calls or new searches."""
    cached_data = ("Cached weather", "http://example.com/cached", {"search_id": 789})
    mock_weather_cache.get.return_value = cached_data

    monkeypatch.setattr("utils.http_client.HTTPClientManager.get_search_client", lambda: mock_http_client)


    weather, source, search_id = await WeatherService.get_weather("Test City", mock_search_service)

    assert weather == "Cached weather"
    assert source == "http://example.com/cached"
    assert search_id == 789
    mock_weather_cache.get.assert_called_once_with("weather", "Test City")
    mock_weather_cache.set.assert_not_called()
    mock_http_client.get.assert_not_called()
    mock_search_service.perform_search.assert_not_called()

def test_format_weather_data_formats_correctly():
    """_format_weather_data should correctly format raw API response into a readable string."""
    api_data = DEFAULT_WEATHER_RESPONSE
    api_data["main"]["temp"] = 15.6
    api_data["main"]["feels_like"] = 14.8
    api_data["wind"] = {"speed": 3.1} # m/s
    api_data["sys"] = {"country": "ZZ"}
    api_data["name"] = "Capital City"

    formatted_string = WeatherService._format_weather_data(api_data)
    assert "Current Weather in Capital City, ZZ" in formatted_string
    assert "- Conditions: Clear (Clear sky)" in formatted_string
    assert "- Temperature: 16°C (feels like 15°C)" in formatted_string
    assert "- Humidity: 60%" in formatted_string
    assert "- Wind Speed: 11 km/h" in formatted_string
