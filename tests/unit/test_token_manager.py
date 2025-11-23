import pytest
from utils.token_manager import TokenManager

@pytest.fixture
def mock_token_manager_config(monkeypatch):
    """Mocks configuration values for TokenManager methods."""
    monkeypatch.setattr(TokenManager, "SAFETY_BUFFER", 0.75)
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: 2000)
    monkeypatch.setattr(TokenManager, "calculate_messages_tokens", lambda messages: len(str(messages).split()))

@pytest.mark.parametrize("model_limit, safety_buffer, expected_safe_limit", [
    (2000, 0.75, 1500),
    (1000, 0.5, 500),
    (5000, 0.8, 4000),
])
def test_check_context_limit_calculates_safe_limit_correctly(monkeypatch, model_limit, safety_buffer, expected_safe_limit):
    """Given various model limits and safety buffers, check_context_limit should calculate the safe limit accurately."""
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: model_limit)
    monkeypatch.setattr(TokenManager, "SAFETY_BUFFER", safety_buffer)
    monkeypatch.setattr(TokenManager, "calculate_messages_tokens", lambda messages: 6) # Mock to return 6 for "short message"

    messages = [{"role": "user", "content": "short message"}]

    within_limit, tokens_used, safe_limit, model_max = TokenManager.check_context_limit(messages, "test-model")

    assert within_limit is True
    assert tokens_used == 6
    assert safe_limit == expected_safe_limit
    assert model_max == model_limit

def test_truncate_history_with_zero_budget_returns_empty_history(mock_token_manager_config, monkeypatch):
    """Given zero budget for history, truncate_history_to_fit should return an empty list for history."""
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: 10)
    monkeypatch.setattr(TokenManager, "calculate_messages_tokens", lambda messages: 100)

    history = [{"role": "user", "content": "long message"}]
    current_prompt = "short prompt"

    truncated_history, included_count = TokenManager.truncate_history_to_fit(
        system_prompt="sys",
        user_memory="",
        current_prompt=current_prompt,
        history=history,
        model_name="test-model",
        additional_reserve=0,
    )

    assert truncated_history == []
    assert included_count == 0

@pytest.mark.parametrize("text, expected_min_tokens", [
    ("hello world", 2), # "hello" and "world"
    ("", 0),
    ("  ", 0),
])
def test_estimate_tokens_with_various_inputs(mock_token_manager_config, text, expected_min_tokens):
    """Given various text inputs, estimate_tokens should provide a reasonable token count."""
    tokens = TokenManager.estimate_tokens(text)
    assert tokens >= expected_min_tokens

def test_truncate_history_to_fit_keeps_newest_messages(mock_token_manager_config, monkeypatch):
    """Given history that exceeds limits, truncate_history_to_fit should prioritize keeping the newest messages."""
    monkeypatch.setattr(TokenManager, "get_model_context_limit", lambda model: 2000)
    
    # Mock token calculation for predictable test
    def mock_calc_tokens_for_truncate(messages):
        if isinstance(messages, str):
            return len(messages.split())
        return sum(len(msg["content"].split()) for msg in messages)

    monkeypatch.setattr(TokenManager, "calculate_messages_tokens", mock_calc_tokens_for_truncate)

    history = [
        {"role": "user", "content": "first message " * 100},  # ~100 tokens
        {"role": "assistant", "content": "second message " * 100}, # ~100 tokens
        {"role": "user", "content": "third message " * 100},   # ~100 tokens
    ]

    truncated_history, included_count = TokenManager.truncate_history_to_fit(
        system_prompt="System prompt with 4 tokens",
        user_memory="",
        current_prompt="What is happening? (4 tokens)",
        history=history,
        model_name="test-model",
        additional_reserve=650,
    )

    truncated_history, included_count = TokenManager.truncate_history_to_fit(
        system_prompt="System prompt",
        user_memory="",
        current_prompt="What is happening?",
        history=history,
        model_name="test-model",
        additional_reserve=689,
    )

    assert included_count == 1
    assert truncated_history == history[-1:]
