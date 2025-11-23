
from unittest.mock import patch, AsyncMock
from models.chat_models import FlowAction, FlowStep, SearchResult
from tests.helpers import assert_sse_event

def test_chat_endpoint_without_api_key_fails(configured_app):
    """Given no API key, when the /chat endpoint is called, it should return a 401 Unauthorized error."""
    response = configured_app.post("/chat", json={"model": "llama", "prompt": "Hello"})
    assert response.status_code == 401

def test_chat_endpoint_with_invalid_api_key_fails(configured_app):
    """Given an invalid API key, when the /chat endpoint is called, it should return a 403 Forbidden error."""
    response = configured_app.post(
        "/chat",
        json={"model": "llama", "prompt": "Hello"},
        headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 403

@patch("services.chat_service.ChatService.orchestrate_chat_flow")
def test_chat_endpoint_returns_json_response(mock_orchestrate_flow, configured_app, auth_headers):
    """Given a valid request, when the /chat endpoint is called, it should return a complete JSON response."""
    async def fake_flow(context):
        yield FlowStep(
            action=FlowAction.RETURN_RESPONSE,
            response="Hi there",
            search_result=SearchResult(performed=False),
            messages=context.messages
        )
    mock_orchestrate_flow.side_effect = fake_flow

    response = configured_app.post(
        "/chat",
        json={"model": "llama", "prompt": "Hello"},
        headers=auth_headers
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "Hi there"
    assert payload["model"] == "llama"
    mock_orchestrate_flow.assert_called_once()

@patch("services.chat_service.ChatService.orchestrate_chat_flow")
def test_chat_stream_endpoint_streams_events(mock_orchestrate_flow, configured_app, auth_headers):
    """Given a valid request, when the /chat/stream endpoint is called, it should stream SSE events."""
    async def fake_flow(context):
        # Yield a search step
        yield FlowStep(action=FlowAction.SEARCH, search_query="test query", search_type="google")
        # Yield the final response
        yield FlowStep(
            action=FlowAction.RETURN_RESPONSE,
            response="streamed response",
            search_result=SearchResult(performed=True, search_id=42),
            messages=context.messages,
        )
    mock_orchestrate_flow.side_effect = fake_flow

    with configured_app.stream("POST", "/chat/stream", json={"model": "llama", "prompt": "query"}, headers=auth_headers) as response:
        response.raise_for_status()
        body = "".join(chunk for chunk in response.iter_text())
    assert_sse_event(body, "status", stage="initializing")
    assert_sse_event(body, "status", stage="searching", message="Searching google for: test query")
    assert_sse_event(body, "done", full_response="streamed response", search_performed=True, search_id=42, search_type=None, search_query=None)

def test_list_models_endpoint_returns_models(configured_app, auth_headers, mocker):
    """Given a configured app, when the /list endpoint is called, it should return available models."""
    mock_ollama_client_instance = AsyncMock()
    mock_ollama_client_instance.list.return_value = {"models": [{"model": "llama3.2:3b"}, {"model": "mistral:7b"}]}
    mocker.patch("routes.models_route.ollama.AsyncClient", return_value=mock_ollama_client_instance)
    
    response = configured_app.get("/list", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"models": ["llama3.2:3b", "mistral:7b"]}
