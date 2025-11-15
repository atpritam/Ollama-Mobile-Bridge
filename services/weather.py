"""
Weather service for retrieving current weather information.
Uses OpenWeatherMap API with fallback to web scraping.
"""
from typing import Tuple, Optional
import httpx

from config import Config
from utils.logger import app_logger


class WeatherService:
    """Service for fetching weather data."""

    @staticmethod
    async def get_weather(city: str) -> Tuple[str, Optional[str]]:
        """
        Get current weather for a city using OpenWeatherMap API.
        Falls back to web scraping if the API key is not available.

        Args:
            city: City name to get weather for

        Returns:
            Tuple of (weather_info, source_url)
        """
        if Config.OPENWEATHER_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=Config.WEATHER_TIMEOUT) as client:
                    response = await client.get(
                        Config.OPENWEATHER_URL,
                        params={
                            "q": city,
                            "appid": Config.OPENWEATHER_API_KEY,
                            "units": "metric"  # Celsius
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        weather_info = WeatherService._format_weather_data(data)
                        app_logger.info(f"Weather data retrieved for {city}")
                        return weather_info, "https://openweathermap.org"

                    elif response.status_code == 404:
                        return f"Weather query: '{city}'\nCity not found. Please check the spelling.", None
                    elif response.status_code == 401:
                        app_logger.warning("Invalid OpenWeather API key, falling back to web scraping")
                    else:
                        app_logger.warning(f"OpenWeather API error (status {response.status_code}), falling back to web scraping")

            except Exception as e:
                app_logger.error(f"Weather API failed: {e}, falling back to web scraping")

        app_logger.info(f"Using web scraping for weather: {city}")
        from services.search import SearchService
        return await SearchService.perform_search("google", f"{city} weather today")

    @staticmethod
    def _format_weather_data(data: dict) -> str:
        """Format weather API response into readable text."""
        weather_desc = data["weather"][0]["description"].capitalize()
        temp = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        humidity = data["main"]["humidity"]
        wind_speed = round(data["wind"]["speed"] * 3.6)

        return f"""Current Weather in {data['name']}, {data['sys']['country']}:
- Conditions: {weather_desc}
- Temperature: {temp}°C (feels like {feels_like}°C)
- Humidity: {humidity}%
- Wind Speed: {wind_speed} km/h"""