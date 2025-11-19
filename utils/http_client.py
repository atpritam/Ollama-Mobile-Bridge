"""
HTTP client utilities with connection pooling.
Provides reusable httpx clients for better performance.
"""
import httpx
from config import Config


class HTTPClientManager:
    """Manages shared httpx clients with connection pooling."""

    _search_client: httpx.AsyncClient | None = None
    _general_client: httpx.AsyncClient | None = None

    @classmethod
    def get_search_client(cls) -> httpx.AsyncClient:
        """
        Get or create a shared httpx client for search operations.

        Features:
        - Connection pooling (reuses TCP connections)
        - Automatic redirect following

        Returns:
            Configured httpx.AsyncClient for search operations
        """
        if cls._search_client is None:
            limits = httpx.Limits(
                max_connections=Config.MAX_CONCURRENT_SCRAPES,
                max_keepalive_connections=5,
                keepalive_expiry=30.0
            )

            cls._search_client = httpx.AsyncClient(
                timeout=Config.SEARCH_TIMEOUT,
                follow_redirects=True,
                max_redirects=Config.MAX_REDIRECTS,
                limits=limits,
                http2=True
            )

        return cls._search_client

    @classmethod
    def get_general_client(cls) -> httpx.AsyncClient:
        """
        Get or create a shared httpx client for general web scraping.

        Features:
        - Connection pooling (reuses TCP connections)
        - Configured with scraping-specific timeouts

        Returns:
            Configured httpx.AsyncClient for web scraping
        """
        if cls._general_client is None:
            limits = httpx.Limits(
                max_connections=Config.MAX_CONCURRENT_SCRAPES * 2,
                max_keepalive_connections=10,
                keepalive_expiry=60.0
            )

            cls._general_client = httpx.AsyncClient(
                timeout=Config.WEB_SCRAPING_TIMEOUT,
                follow_redirects=True,
                max_redirects=Config.MAX_REDIRECTS,
                limits=limits,
                http2=True
            )

        return cls._general_client

    @classmethod
    async def close_all(cls) -> None:
        """
        Close all managed clients and clean up connections.
        """
        if cls._search_client is not None:
            await cls._search_client.aclose()
            cls._search_client = None

        if cls._general_client is not None:
            await cls._general_client.aclose()
            cls._general_client = None