import pytest
from unittest.mock import patch, AsyncMock

from config import Config
from models.chat_models import SearchResult, FlowAction, FlowStep
from services.chat_service import ChatService
from utils.constants import SearchType
from utils.token_manager import TokenManager


@pytest.fixture
def mock_token_manager(monkeypatch):
    """Fixture to mock TokenManager methods."""
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: 8000)
    monkeypatch.setattr(
        TokenManager,
        "check_context_limit",
        lambda messages, model: (True, 120, 400, 8000)
    )

@pytest.mark.parametrize("query, expected", [
    ("What is the latest NASA news this week?", True),
    ("Explain the theory of relativity.", False),
    ("current price of bitcoin", True),
])
def test_preflight_search_check_detects_temporal_signals(query, expected):
    """Given a query, when preflight_search_check is called, then it should detect temporal signals correctly."""
    assert ChatService.preflight_search_check(query) is expected

@pytest.mark.parametrize("response, expected", [
    ("I don't have up-to-date information after my training data in 2023.", True),
    ("As of my last update in 2022", True),
    ("Here's what I know from earlier research.", False),
])
def test_detect_knowledge_cutoff_matches_patterns(response, expected):
    """Given a response, when detect_knowledge_cutoff is called, then it should identify cutoff phrases."""
    assert ChatService.detect_knowledge_cutoff(response) is expected

@pytest.mark.parametrize("text, expected_type, expected_query", [
    ("GOOGLE: latest mars mission timeline", SearchType.GOOGLE, "latest mars mission timeline"),
    ("SEARCH: opinions on new gpus", "NEEDS_QUERY_EXTRACTION", "opinions on new gpus"),
    ("Tell me about the moon.", None, None),
])
def test_parse_search_command(text, expected_type, expected_query):
    """Given a text, when _parse_search_command is called, it should return the correct search type and query."""
    search_type, search_query = ChatService._parse_search_command(text)
    assert search_type == expected_type
    assert search_query == expected_query

def test_parse_recall_command_extracts_id():
    """Given a recall command, when _parse_recall_command is called, it should extract the ID."""
    assert ChatService._parse_recall_command("RECALL: 7") == 7

def test_clean_response_removes_search_tags():
    """Given a noisy response, when clean_response is called, it should remove search tags."""
    assert ChatService.clean_response("GOOGLE: latest launch\nHere is the answer.") == "Here is the answer."

def test_strip_search_id_tag_removes_all_positions():
    """Given a text with search ID tags, when strip_search_id_tag is called, it should remove all occurrences."""
    text = "[search_id: 5] Details about the topic [search_id: 5]"
    assert ChatService.strip_search_id_tag(text) == "Details about the topic"

@pytest.mark.anyio
async def test_yield_search_and_response_sequence(chat_context, mock_token_manager):
    """Given a search result, when _yield_search_and_response is called, it should yield search and response steps."""
    search_result = SearchResult(
        performed=True,
        search_type=SearchType.GOOGLE,
        search_query="latest launch update",
        search_results="Launch scheduled for tomorrow.",
        source_url="https://example.com",
        search_id=10
    )

    steps = [step async for step in ChatService._yield_search_and_response(search_result, chat_context)]

    assert len(steps) == 2
    assert steps[0].action == FlowAction.SEARCH
    assert steps[0].search_query == "latest launch update"
    assert steps[1].action == FlowAction.STREAM_RESPONSE
    assert "Launch scheduled for tomorrow." in steps[1].messages[0]["content"]

def test_build_response_metadata_includes_search_details(chat_request, mock_token_manager):
    """Given a search result, when build_response_metadata is called, it should include all relevant details."""
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Tell me something"}]
    search_result = SearchResult(
        performed=True,
        search_type=SearchType.GOOGLE,
        search_query="latest launch update",
        source_url="https://example.com/article",
        search_id=23
    )

    metadata = ChatService.build_response_metadata(chat_request, messages, search_result)

    assert metadata["search_performed"] is True
    assert metadata["search_type"] == SearchType.GOOGLE
    assert metadata["search_id"] == 23
    assert metadata["tokens"]["used"] == 120
    assert metadata["tokens"]["limit"] == 400

@pytest.mark.anyio
async def test_orchestration_for_small_model_with_preflight_triggers_search(chat_context, monkeypatch):
    """Given a small model and a temporal query, when orchestrate_chat_flow is called, it should trigger a search."""
    monkeypatch.setattr(Config, "is_small_model", classmethod(lambda cls, model: True))
    monkeypatch.setattr(ChatService, "preflight_search_check", staticmethod(lambda prompt: True))
    monkeypatch.setattr(ChatService, "extract_search_query", AsyncMock(return_value=SearchResult(performed=True, search_type=SearchType.GOOGLE, search_query="latest update", search_results="fresh info")))

    mock_generator_steps = [
        FlowStep(action=FlowAction.SEARCH, search_query="latest update"),
        FlowStep(action=FlowAction.STREAM_RESPONSE)
    ]

    async def mock_yield_search_and_response_gen(*args, **kwargs):
        for step in mock_generator_steps:
            yield step

    monkeypatch.setattr(ChatService, "_yield_search_and_response", mock_yield_search_and_response_gen)

    steps = [step async for step in ChatService.orchestrate_chat_flow(chat_context)]
    assert [step.action for step in steps] == [FlowAction.SEARCH, FlowAction.STREAM_RESPONSE]

@pytest.mark.anyio
async def test_orchestration_for_large_model_bypasses_preflight(chat_context, monkeypatch):
    """Given a large model, when orchestrate_chat_flow is called, it should proceed to the main LLM response processing."""
    monkeypatch.setattr(Config, "is_small_model", classmethod(lambda cls, model: False))

    mock_generator_steps = [FlowStep(action=FlowAction.RETURN_RESPONSE)]

    async def mock_process_llm_response_gen(*args, **kwargs):
        for step in mock_generator_steps:
            yield step

    monkeypatch.setattr(ChatService, "_process_llm_response", mock_process_llm_response_gen)

    steps = [step async for step in ChatService.orchestrate_chat_flow(chat_context)]
    assert steps[0].action == FlowAction.RETURN_RESPONSE
@pytest.mark.anyio
async def test_recall_from_cache_when_hit(mock_cache):
    """Given a RECALL command with a valid ID, when detect_and_recall_from_cache is called, it should return cached results."""
    mock_cache.get_by_id.return_value = ("cached results", "https://example.com", {"search_type": SearchType.GOOGLE, "query": "mars news"})
    with patch("utils.cache.get_search_cache", lambda: mock_cache):
        detected, recall_id, result = await ChatService.detect_and_recall_from_cache("RECALL: 42")
        assert detected and recall_id == 42
        assert result.performed and result.search_query == "mars news"

@pytest.mark.anyio
async def test_recall_from_cache_when_miss(mock_cache):
    """Given a RECALL command with an invalid ID, when detect_and_recall_from_cache is called, it should return a miss."""
    mock_cache.get_by_id.return_value = None
    with patch("utils.cache.get_search_cache", lambda: mock_cache):
        detected, recall_id, result = await ChatService.detect_and_recall_from_cache("RECALL: 7")
        assert detected and recall_id == 7
        assert not result.performed
