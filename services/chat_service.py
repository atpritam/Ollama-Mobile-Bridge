"""
Chat service containing core chat processing logic.
Handles search detection, query extraction, and chat flow orchestration.
"""
import re
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import urlparse

from models.api_models import ChatRequest
from models.chat_models import ChatContext, SearchResult, FlowAction, FlowStep
from utils.constants import (
    DEFAULT_SYSTEM_PROMPT,
    SIMPLE_SYSTEM_PROMPT,
    SEARCH_RESULT_SYSTEM_PROMPT,
    SEARCH_QUERY_EXTRACTION_PROMPT,
    SearchType,
    Patterns
)
from config import Config
from services.weather import WeatherService
from services.search import SearchService
from utils.logger import app_logger
from utils.token_manager import TokenManager


class ChatService:
    """Service for handling chat logic."""

    SEARCH_TYPE_MAPPING = {
        'WEATHER': SearchType.WEATHER,
        'GOOGLE': SearchType.GOOGLE,
        'REDDIT': SearchType.REDDIT,
        'WIKI': SearchType.WIKIPEDIA
    }

    @staticmethod
    def preflight_search_check(user_query: str) -> bool:
        """Pre-flight check: Detect recency/realtime indicators."""
        query_lower = user_query.lower()

        # Temporal indicators
        temporal_keywords = ["2025", "2026", "latest", "recent", "current",
            "today", "yesterday", "this week", "this month", "this year",
            "now", "right now", "breaking", "last week", "last month"
        ]

        # Real-time data keywords
        realtime_keywords = [
            "weather", "temperature", "forecast", "stock", "price",
            "news", "election", "score", "won", "elected", "winner",
            "happening", "live", "update"
        ]

        has_temporal = any(kw in query_lower for kw in temporal_keywords)
        has_realtime = any(kw in query_lower for kw in realtime_keywords)

        if (has_temporal and has_realtime) or has_temporal:
            app_logger.info(f"Pre-flight: Recency pattern detected in query")
            return True

        return False

    @staticmethod
    def detect_knowledge_cutoff(response: str) -> bool:
        """Detect if model response mentions knowledge cutoff or lack of current info."""
        response_lower = response.lower()

        for pattern in Patterns.KNOWLEDGE_CUTOFF_PATTERNS:
            if re.search(pattern, response_lower):
                app_logger.info(f"Knowledge cutoff detected: pattern '{pattern}' matched")
                return True

        return False

    @staticmethod
    async def execute_search(context: ChatContext, search_type: str, search_query: str) -> tuple[str, str]:
        """Execute search based on type."""
        search_service = SearchService(context)

        if search_type == SearchType.WEATHER:
            return await WeatherService.get_weather(search_query)
        elif search_type == SearchType.REDDIT:
            return await search_service.perform_search(SearchType.REDDIT, f"reddit {search_query}")
        elif search_type == SearchType.WIKIPEDIA:
            return await search_service.perform_search(SearchType.WIKIPEDIA, f"wikipedia {search_query}")
        else:
            return await search_service.perform_search(search_type, search_query)

    @staticmethod
    def parse_search_type(search_type_raw: str) -> str:
        """Parse and map search type string to internal type constant."""
        return ChatService.SEARCH_TYPE_MAPPING.get(search_type_raw.upper(), SearchType.GOOGLE)

    @staticmethod
    def clean_response(response: str) -> str:
        """Clean response by removing SEARCH tags and extra whitespace."""
        clean = re.sub(
            Patterns.SEARCH_TAG_CLEANUP,
            '',
            response,
            flags=re.IGNORECASE
        ).strip()

        return clean or response

    @staticmethod
    async def extract_search_query(context: ChatContext, original_response: str) -> SearchResult:
        """
        Use specialized prompt to extract search query from user question.
        This is called when the model mentioned knowledge cutoff or when pre-flight triggers.
        """
        extraction_system_prompt = SEARCH_QUERY_EXTRACTION_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

        extraction_messages = [
            {"role": "system", "content": extraction_system_prompt}
        ]

        if context.request.history:
            limited_history = context.request.history[-Config.MAX_HISTORY_MESSAGES:]
            extraction_messages.extend([
                {"role": msg.role, "content": msg.content}
                for msg in limited_history
            ])

        extraction_messages.append({
            "role": "user",
            "content": context.prompt
        })

        call_num = context.next_call_number()
        app_logger.info(f"LLM Call #{call_num}: Extracting search query")

        response = await context.client.chat(
            model=context.model_name,
            messages=extraction_messages
        )

        extraction_response = response['message']['content'].strip()
        app_logger.info(f"LLM Call #{call_num} response: {extraction_response}")

        search_type, search_query = ChatService._parse_search_command(extraction_response)

        if not search_type:
            app_logger.warning(f"Extraction failed, using original query")
            search_type = SearchType.GOOGLE
            search_query = context.prompt
        else:
            app_logger.info(f"Extraction successful: {search_type} - '{search_query}'")

        search_results, source_url = await ChatService.execute_search(context, search_type, search_query)

        return SearchResult(
            performed=True,
            search_type=search_type,
            search_query=search_query,
            search_results=search_results,
            source_url=source_url
        )

    @staticmethod
    def get_system_prompt(request: ChatRequest) -> str:
        """Get system prompt from request or use default."""
        if request.system_prompt:
            return request.system_prompt

        user_context = ChatService._format_user_context(request.user_memory)

        if Config.is_small_model(request.model):
            return SIMPLE_SYSTEM_PROMPT.format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                user_context=user_context
            )

        return DEFAULT_SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            user_context=user_context
        )

    @staticmethod
    def prepare_messages(request: ChatRequest, system_prompt: str) -> list:
        """
        Prepare a messages list for LLM with system prompt and history.
        Uses intelligent token-based truncation to fit within the model's context window.
        """
        messages = []

        messages.append({"role": "system", "content": system_prompt})

        # Truncate history to fit within token budget
        if request.history:
            truncated_history, messages_included = TokenManager.truncate_history_to_fit(
                system_prompt=system_prompt,
                user_memory=request.user_memory or "",
                current_prompt=request.prompt,
                history=[{"role": msg.role, "content": msg.content} for msg in request.history],
                model_name=request.model
            )
            messages.extend(truncated_history)

        messages.append({"role": "user", "content": request.prompt})

        # validation
        within_limit, tokens_used, safe_limit, model_max = TokenManager.check_context_limit(
            messages, request.model
        )

        if not within_limit:
            raise ValueError(
                f"Context limit exceeded. "
                f"Tokens: {tokens_used}/{safe_limit} (model max: {model_max}). "
            )

        return messages

    @staticmethod
    async def detect_and_perform_search(assistant_response: str, context: ChatContext) -> SearchResult:
        """Detect if LLM requested a search and perform it.

        Args:
            assistant_response: Response from LLM
            context: ChatContext containing model info and other request data

        Returns:
            SearchResult object containing search information
        """
        search_type, search_query = ChatService._parse_search_command(assistant_response)

        if not search_type:
            return SearchResult(performed=False)

        app_logger.info(f"Search triggered: {search_type} - '{search_query}'")

        search_results, source_url = await ChatService.execute_search(context, search_type, search_query)
        app_logger.debug(f"Search results: {search_results[:200]}")

        return SearchResult(
            performed=True,
            search_type=search_type,
            search_query=search_query,
            search_results=search_results,
            source_url=source_url
        )

    @staticmethod
    def extract_domain(url: str) -> str | None:
        """Extract domain from URL."""
        if not url:
            return None
        parsed = urlparse(url)
        return parsed.netloc

    @staticmethod
    def _format_user_context(user_memory: str | None) -> str:
        """Format user memory into context string."""
        if not user_memory:
            return ""
        user_context = user_memory.strip()
        return f"User Context: [{user_context}]" if user_context else ""

    @staticmethod
    def _parse_search_command(text: str) -> tuple[str | None, str | None]:
        """Parse search command from text and extract type and query.

        Returns:
            Tuple of (search_type, search_query) or (None, None) if no match
        """
        search_match = re.search(Patterns.SEARCH_WITH_TYPE, text, re.IGNORECASE)

        if search_match:
            search_type_raw = search_match.group(1).upper()
            search_query = search_match.group(2).strip()
            search_query = search_query.strip('"').strip("'")
            search_type = ChatService.parse_search_type(search_type_raw)
            return search_type, search_query

        # Try fallback pattern
        fallback_match = re.search(Patterns.SEARCH_FALLBACK, text, re.IGNORECASE)
        if fallback_match:
            search_query = fallback_match.group(1).strip()
            return SearchType.GOOGLE, search_query

        return None, None

    @staticmethod
    def _build_messages(request: ChatRequest,system_prompt: str, additional_reserve: int = 0, validate: bool = True) -> list:
        """Build messages list with system prompt, history, and user prompt.

        Args:
            request: Chat request containing prompt and history
            system_prompt: System prompt to use
            additional_reserve: Additional tokens to reserve for search results
            validate: Whether to validate context limits and raise on overflow

        Returns:
            List of messages ready for LLM
        """
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # Truncate history to fit within token budget
        if request.history:
            truncated_history, messages_included = TokenManager.truncate_history_to_fit(
                system_prompt=system_prompt,
                user_memory=request.user_memory or "",
                current_prompt=request.prompt,
                history=[{"role": msg.role, "content": msg.content} for msg in request.history],
                model_name=request.model,
                additional_reserve=additional_reserve
            )
            messages.extend(truncated_history)

        messages.append({"role": "user", "content": request.prompt})

        # Validation
        if validate:
            within_limit, tokens_used, safe_limit, model_max = TokenManager.check_context_limit(
                messages, request.model
            )

            if not within_limit:
                raise ValueError(
                    f"Context limit exceeded. "
                    f"Tokens: {tokens_used}/{safe_limit} (model max: {model_max}). "
                )

        return messages

    @staticmethod
    def _prepare_search_response_messages(context: ChatContext, search_results: str) -> list:
        """Prepare messages with search results injected into system prompt."""
        search_system_prompt = SEARCH_RESULT_SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            user_context=ChatService._format_user_context(context.request.user_memory),
            search_results=search_results
        )

        # Calculate search result token cost
        search_result_tokens = TokenManager.estimate_tokens(search_results)
        app_logger.info(f"Search results size: {search_result_tokens} tokens")

        return ChatService._build_messages(
            request=context.request,
            system_prompt=search_system_prompt,
            additional_reserve=search_result_tokens,
            validate=False  # Just warn, don't throw
        )

    @staticmethod
    async def _yield_search_and_response(search_result: SearchResult,context: ChatContext) -> AsyncIterator[FlowStep]:
        """
        Yield search action followed by stream response with search results.

        Args:
            search_result: SearchResult object with search data
            context: ChatContext with request data

        Yields:
            FlowStep for search action and stream response
        """
        # Yield search action
        yield FlowStep(
            action=FlowAction.SEARCH,
            search_type=search_result.search_type,
            search_query=search_result.search_query
        )

        # Prepare final messages with search results
        final_messages = ChatService._prepare_search_response_messages(
            context,
            search_result.search_results
        )

        # Yield stream response action
        yield FlowStep(
            action=FlowAction.STREAM_RESPONSE,
            messages=final_messages,
            search_result=search_result
        )

    @staticmethod
    async def _handle_search_tag_detected(search_result: SearchResult,context: ChatContext) -> AsyncIterator[FlowStep]:
        """Handle flow when SEARCH tag is detected in LLM response."""
        app_logger.info("SEARCH tag detected")
        async for step in ChatService._yield_search_and_response(search_result, context):
            yield step

    @staticmethod
    async def _handle_knowledge_cutoff(context: ChatContext,assistant_response: str) -> AsyncIterator[FlowStep]:
        """Handle flow when knowledge cutoff is detected in LLM response."""
        app_logger.info("Cutoff detected, triggering search query extraction")

        # Extract search query
        search_result = await ChatService.extract_search_query(context, assistant_response)

        # Yield search and response
        async for step in ChatService._yield_search_and_response(search_result, context):
            yield step

    @staticmethod
    async def _process_llm_response(context: ChatContext, call_num: int) -> AsyncIterator[FlowStep]:
        """Process LLM response and handle search/cutoff detection.

        Args:
            context: ChatContext with request data
            call_num: LLM call number for logging

        Yields:
            FlowStep for various actions (search, response, etc.)
        """
        response = await context.client.chat(
            model=context.request.model,
            messages=context.messages
        )
        assistant_response = response['message']['content']
        app_logger.debug(f"LLM Call #{call_num} response preview: {assistant_response[:100]}...")

        # Check for SEARCH tag in response
        search_result = await ChatService.detect_and_perform_search(assistant_response, context)

        if search_result.performed:
            async for step in ChatService._handle_search_tag_detected(search_result, context):
                yield step
            return

        # Check for knowledge cutoff pattern
        if ChatService.detect_knowledge_cutoff(assistant_response):
            async for step in ChatService._handle_knowledge_cutoff(context, assistant_response):
                yield step
            return

        # No search needed, return Call #1 response
        clean_response = ChatService.clean_response(assistant_response)
        yield FlowStep(
            action=FlowAction.RETURN_RESPONSE,
            response=clean_response,
            messages=context.messages,
            search_result=SearchResult(performed=False),
            call_number=call_num
        )

    @staticmethod
    async def orchestrate_chat_flow(context: ChatContext) -> AsyncIterator[FlowStep]:
        """Core flow orchestrator that yields decision points."""
        is_small = Config.is_small_model(context.model_name)

        if is_small:
            # SMALL MODEL - Scenario A: Pre-flight detects a recency pattern
            if ChatService.preflight_search_check(context.prompt):
                # Call #1: Generate proper search query using specialized prompt
                search_result = await ChatService.extract_search_query(context, "")
                async for step in ChatService._yield_search_and_response(search_result, context):
                    yield step
                return

        # SMALL MODEL - Scenario B OR LARGE MODEL: Initial call with prompt
        call_num = context.next_call_number()
        log_prefix = "Pre-flight passed, Attempting" if is_small else "Attempting"
        app_logger.info(f"LLM Call #{call_num}: {log_prefix} First Call")

        # Process response and handle search/cutoff detection
        async for step in ChatService._process_llm_response(context, call_num):
            yield step

    @staticmethod
    def build_response_metadata(request: ChatRequest,messages: list,search_result: SearchResult) -> dict:
        """Build response metadata dictionary."""
        # Calculate token usage
        _, tokens_used, safe_limit, model_max = TokenManager.check_context_limit(
            messages, request.model
        )

        metadata = {
            "model": request.model,
            "context_messages_count": len(messages) - 1,
            "search_performed": search_result.performed,
            "tokens": {
                "used": tokens_used,
                "limit": safe_limit,
                "model_max": model_max,
                "usage_percent": round((tokens_used / safe_limit) * 100, 1)
            }
        }

        if search_result.performed:
            metadata["search_type"] = search_result.search_type
            metadata["search_query"] = search_result.search_query

            if search_result.source_url:
                metadata["source"] = search_result.source_url

        return metadata