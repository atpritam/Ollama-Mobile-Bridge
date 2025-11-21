import os
import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse

async def root(request):
    return PlainTextResponse("API Running")

@pytest.fixture
def app_with_middleware(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda key, default=None: "valid-key" if key == "API_KEY" else default)
    from auth import APIKeyMiddleware
    APIKeyMiddleware.API_KEY = os.getenv("API_KEY", "") 

    app = Starlette()
    app.add_middleware(APIKeyMiddleware)
    app.add_route("/", root)
    return app

@pytest.fixture
def client(app_with_middleware):
    return TestClient(app_with_middleware)

@pytest.fixture
def client(app_with_middleware):
    return TestClient(app_with_middleware)

@pytest.mark.parametrize("header_name", ["X-API-Key", "x-api-key", "X-Api-Key"])
def test_api_key_middleware_accepts_case_insensitive_header(client, header_name):
    """Given a valid API key, when the API key header is provided with different casings, it should be accepted."""
    response = client.get("/", headers={header_name: "valid-key"})
    assert response.status_code == 200
    assert response.text == "API Running"

def test_api_key_middleware_rejects_missing_header(client):
    """Given a missing API key header, when accessing a protected route, it should return 401 Unauthorized."""
    response = client.get("/")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key. Include 'X-API-Key' header in your request."
    assert response.json()["error"] == "unauthorized"

def test_api_key_middleware_rejects_invalid_key(client):
    """Given an invalid API key, when accessing a protected route, it should return 403 Forbidden."""
    response = client.get("/", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key"
    assert response.json()["error"] == "forbidden"

def test_api_key_middleware_handles_malformed_header(client):
    """Given a malformed API key header (e.g., empty), when accessing a protected route, it should return 401 Unauthorized."""
    response = client.get("/", headers={"X-API-Key": ""})
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API key. Include 'X-API-Key' header in your request."
    assert response.json()["error"] == "unauthorized"
