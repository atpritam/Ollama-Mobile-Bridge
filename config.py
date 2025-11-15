"""
Configuration module for the Ollama Mobile Bridge application.
Handles environment variables and application settings.
"""
import os
import re
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration class."""

    # API Keys
    BRAVE_SEARCH_API_KEY: str = os.getenv("BRAVE_SEARCH_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")

    # API Configuration
    BRAVE_SEARCH_URL: str = "https://api.search.brave.com/res/v1/web/search"
    OPENWEATHER_URL: str = "https://api.openweathermap.org/data/2.5/weather"

    # Application Settings
    APP_TITLE: str = "Ollama Mobile Bridge"
    MAX_HISTORY_MESSAGES: int = 30
    DEFAULT_SEARCH_RESULTS_COUNT: int = 5

    # Timeouts (in seconds)
    SEARCH_TIMEOUT: float = 15.0
    WEATHER_TIMEOUT: float = 10.0
    WEB_SCRAPING_TIMEOUT: float = 10.0

    # Content web Limits (default)
    MAX_HTML_TEXT_LENGTH: int = 4000

    # Small model detection in b parameters (default)
    SMALL_MODEL_THRESHOLD: float = 4.0

    @classmethod
    def extract_model_param_size(cls, model_name: str) -> float | None:
        """Extract parameter size from model name."""
        pattern = r'(\d+\.?\d*)b'
        match = re.search(pattern, model_name.lower())

        if match:
            return float(match.group(1))

        return None

    @classmethod
    def is_small_model(cls, model_name: str, threshold: float = SMALL_MODEL_THRESHOLD) -> bool:
        """Detect if the model is small (below threshold parameter)."""
        model_lower = model_name.lower()

        if any(keyword in model_lower for keyword in ['tiny', 'mini', 'small']):
            return True

        param_size = cls.extract_model_param_size(model_name)
        if param_size is not None:
            return param_size < threshold

        return False

    @classmethod
    def get_max_html_text_length(cls, model_name: str) -> int:
        """Get the MAX_HTML_TEXT_LENGTH based on model parameter size."""
        param_size = cls.extract_model_param_size(model_name)
        if param_size is not None:
            return 8000 if param_size >= 12 else 4000

        return cls.MAX_HTML_TEXT_LENGTH

    @classmethod
    def validate(cls) -> None:
        """Validate configuration and print warnings for missing API keys."""
        if not cls.BRAVE_SEARCH_API_KEY:
            print("   WARNING: BRAVE_SEARCH_API_KEY not found in .env file")
            print("   Web search will not work. Get your free API key from: https://brave.com/search/api/")

        if not cls.OPENWEATHER_API_KEY:
            print("   WARNING: OPENWEATHER_API_KEY not found in .env file")
            print("   Weather queries will use web scraping fallback. Get free key from: https://openweathermap.org/api")


Config.validate()