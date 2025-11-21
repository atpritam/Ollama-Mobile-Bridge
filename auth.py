"""
Authentication middleware for API key verification.
"""
import os
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import app_logger


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Checks X-API-Key header against configured API_KEY.
    """

    EXCLUDED_PATHS = {"/docs", "/openapi.json", "/redoc"}
    API_KEY: str = os.getenv("API_KEY", "")

    async def dispatch(self, request: Request, call_next):
        """
        Process each request and verify API key.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler or error response
        """
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        if not self.API_KEY:
            app_logger.error("CRITICAL: API_KEY not set in .env file!")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Server misconfiguration: API_KEY not set. Please configure API_KEY in .env file.",
                    "error": "server_error"
                },
            )

        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")

        if not api_key:
            app_logger.warning(
                f"Unauthorized request from {request.client.host} - Missing API key"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Missing API key. Include 'X-API-Key' header in your request.",
                    "error": "unauthorized"
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        if api_key != self.API_KEY:
            app_logger.warning(
                f"Forbidden request from {request.client.host} - Invalid API key"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": "Invalid API key",
                    "error": "forbidden"
                },
            )

        response = await call_next(request)
        return response