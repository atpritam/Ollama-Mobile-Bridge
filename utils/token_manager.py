"""
Token management utilities for context window handling.
Provides token counting and context limit validation.
"""
import ollama
from utils.logger import app_logger
from config import Config


class TokenManager:
    """Manages token counting and context limit validation."""

    # Cache for model context limits
    _context_cache = {}

    # Safety buffer context limit percentage
    SAFETY_BUFFER = Config.SAFETY_BUFFER

    # Token reservation for search results (to prevent context overflow when search results are injected)
    SEARCH_RESULT_RESERVE = 4000

    @staticmethod
    def get_model_context_limit(model_name: str) -> int:
        """Get model's maximum context window from Ollama."""
        if model_name in TokenManager._context_cache:
            return TokenManager._context_cache[model_name]

        try:
            info = ollama.show(model_name)
            model_info = info.modelinfo if hasattr(info, 'modelinfo') else info.get('modelinfo', {})

            for key in model_info:
                if key.endswith('.context_length'):
                    context_limit = model_info[key]
                    TokenManager._context_cache[model_name] = context_limit
                    app_logger.info(f"Model {model_name} context limit: {context_limit:,} tokens")
                    return context_limit

            app_logger.warning(f"Could not find context_length for {model_name}, using default 8192")
            TokenManager._context_cache[model_name] = 8192
            return 8192

        except Exception as e:
            app_logger.error(f"Failed to get context limit for {model_name}: {e}")
            TokenManager._context_cache[model_name] = 8192
            return 8192

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text using character-based approximation."""
        if not text:
            return 0

        char_estimate = len(text) // 4
        word_estimate = len(text.split())

        return int((char_estimate * 0.6) + (word_estimate * 0.4))

    @staticmethod
    def calculate_messages_tokens(messages: list[dict]) -> int:
        """Calculate total tokens for a list of messages."""
        total_tokens = 0

        for msg in messages:
            content = msg.get('content', '')
            total_tokens += TokenManager.estimate_tokens(content)
            total_tokens += 4

        return total_tokens

    @staticmethod
    def check_context_limit(
        messages: list[dict],
        model_name: str
    ) -> tuple[bool, int, int, int]:
        """
        Check if messages fit within model's context limit with safety buffer.

        Args:
            messages: List of message dictionaries
            model_name: Name of the model

        Returns:
            Tuple of (within_limit, tokens_used, safe_limit, model_max)
            - within_limit: True if tokens_used <= safe_limit
            - tokens_used: Estimated tokens in messages
            - safe_limit: Maximum tokens with safety buffer applied
            - model_max: Model's actual maximum context length
        """
        model_max = TokenManager.get_model_context_limit(model_name)
        safe_limit = int(model_max * TokenManager.SAFETY_BUFFER)
        tokens_used = TokenManager.calculate_messages_tokens(messages)
        within_limit = tokens_used <= safe_limit

        app_logger.debug(
            f"Token check: {tokens_used}/{safe_limit} tokens "
            f"(model max: {model_max}, safety: {int(TokenManager.SAFETY_BUFFER * 100)}%)"
        )

        return within_limit, tokens_used, safe_limit, model_max

    @staticmethod
    def truncate_history_to_fit(
        system_prompt: str,
        user_memory: str,
        current_prompt: str,
        history: list[dict],
        model_name: str,
        additional_reserve: int = 0
    ) -> tuple[list[dict], int]:
        """
        Truncate history to fit within context limit while preserving recent messages.

        1. Calculate tokens for system prompt + user memory + current prompt
        2. Calculate remaining budget for history
        3. Include history messages from newest to oldest until budget exhausted

        Args:
            system_prompt: System prompt text
            user_memory: User memory/context text
            current_prompt: Current user prompt
            history: List of historical messages
            model_name: Name of the model
            additional_reserve: Additional tokens to reserve (e.g., for search results)

        Returns:
            Tuple of (truncated_history, messages_included)
        """
        model_max = TokenManager.get_model_context_limit(model_name)
        safe_limit = int(model_max * TokenManager.SAFETY_BUFFER)

        system_tokens = TokenManager.estimate_tokens(system_prompt)
        memory_tokens = TokenManager.estimate_tokens(user_memory) if user_memory else 0
        current_tokens = TokenManager.estimate_tokens(current_prompt)
        fixed_overhead = 12

        fixed_cost = system_tokens + memory_tokens + current_tokens + fixed_overhead

        # Reserve ~500 tokens for model response
        response_reserve = 500

        # Calculate budget for history
        history_budget = safe_limit - fixed_cost - response_reserve - additional_reserve

        if history_budget <= 0:
            app_logger.warning(
                f"No budget for history. Fixed cost: {fixed_cost}, "
                f"Additional reserve: {additional_reserve}, Safe limit: {safe_limit}"
            )
            return [], 0

        # Add messages from newest to oldest
        truncated_history = []
        tokens_used = 0

        for msg in reversed(history):
            msg_tokens = TokenManager.estimate_tokens(msg.get('content', '')) + 4

            if tokens_used + msg_tokens <= history_budget:
                truncated_history.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break

        messages_included = len(truncated_history)
        messages_dropped = len(history) - messages_included

        if messages_dropped > 0:
            reserve_msg = f" (reserved {additional_reserve} for search)" if additional_reserve > 0 else ""
            app_logger.info(
                f"Truncated history: kept {messages_included}/{len(history)} messages "
                f"({tokens_used}/{history_budget} tokens){reserve_msg}"
            )

        return truncated_history, messages_included

    @staticmethod
    def clear_cache():
        """Clear the context limit cache."""
        TokenManager._context_cache.clear()