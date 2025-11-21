"""
Weather service for retrieving current weather information.
Uses OpenWeatherMap API with fallback to web scraping.
"""
from typing import Tuple, Optional
from config import Config
from utils.logger import app_logger
from utils.http_client import HTTPClientManager
from utils.cache import get_search_cache


class WeatherService:
    """Service for fetching weather data."""

    @staticmethod
    async def get_weather(city: str, search_service=None) -> Tuple[str, Optional[str], Optional[int]]:
        """
        Get current weather for a city using OpenWeatherMap API.
        Falls back to web scraping if the API key is not available.

        Args:
            city: City name to get weather for
            search_service: SearchService instance for fallback web scraping

        Returns:
            Tuple of (weather_info, source_url, search_id)
        """
        # Check cache first
        cache = get_search_cache()
        cached = cache.get("weather", city)
        if cached:
            results, source_url, metadata = cached
            search_id = metadata["search_id"]
            app_logger.info(f"Cache HIT: weather/{city} -> {search_id}")
            return results, source_url, search_id

        if Config.OPENWEATHER_API_KEY:
            try:
                client = HTTPClientManager.get_search_client()
                response = await client.get(
                    Config.OPENWEATHER_URL,
                    params={
                        "q": city,
                        "appid": Config.OPENWEATHER_API_KEY,
                        "units": "metric"  # Celsius
                    },
                    timeout=Config.WEATHER_TIMEOUT
                )

                if response.status_code == 200:
                    data = await response.json()
                    weather_info = WeatherService._format_weather_data(data)
                    app_logger.info(f"Weather data retrieved for {city}")

                    city_id = data.get('id')
                    source_url = f"https://openweathermap.org/city/{city_id}" if city_id else "https://openweathermap.org"

                    # Cache the result
                    cache = get_search_cache()
                    scraped_contents = {source_url: weather_info}
                    search_id = cache.set("weather", city, scraped_contents, None)

                    return weather_info, source_url, search_id

                elif response.status_code == 404:
                    error_msg = f"Weather query: '{city}'\nCity not found. Please check the spelling."
                    cache = get_search_cache()
                    scraped_contents = {}  # No content for error
                    search_id = cache.set("weather", city, scraped_contents, error_msg)
                    return error_msg, None, search_id
                elif response.status_code == 401:
                    app_logger.warning("Invalid OpenWeather API key, falling back to web scraping")
                else:
                    app_logger.warning(f"OpenWeather API error (status {response.status_code}), falling back to web scraping")

            except Exception as e:
                app_logger.error(f"Weather API failed: {e}, falling back to web scraping")

        # Fallback to web scraping
        app_logger.info(f"Using web scraping for weather: {city}")
        return await search_service.perform_search("google", f"{city} weather today")

    @staticmethod
    def _format_weather_data(data: dict) -> str:
        """Format weather API response into readable text."""
        weather_main = data["weather"][0]["main"].capitalize()
        weather_desc = data["weather"][0]["description"].capitalize()
        temp = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        humidity = data["main"]["humidity"]
        wind_speed = round(data["wind"]["speed"] * 3.6)

        return f"""Current Weather in {data['name']}, {data['sys']['country']}:
- Conditions: {weather_main} ({weather_desc})
- Temperature: {temp}°C (feels like {feels_like}°C)
- Humidity: {humidity}%
- Wind Speed: {wind_speed} km/h"""