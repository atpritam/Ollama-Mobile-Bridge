"""
Chat service containing core chat processing logic.
Handles search detection, query extraction, and chat flow orchestration.
"""
import re
from datetime import datetime
from typing import AsyncIterator, Optional
from urllib.parse import urlparse

from models.api_models import ChatRequest
from models.chat_models import ChatContext, SearchResult, FlowAction, FlowStep
from utils.constants import (
    DEFAULT_SYSTEM_PROMPT,
    SIMPLE_SYSTEM_PROMPT,
    SEARCH_RESULT_SYSTEM_PROMPT,
    RECALL_SYNTHESIS_PROMPT,
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
        'WIKI': SearchType.WIKIPEDIA,
        'WIKIPEDIA': SearchType.WIKIPEDIA,
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
    async def execute_search(context: ChatContext, search_type: str, search_query: str) -> tuple[str, Optional[str], Optional[int]]:
        """Execute search based on type."""
        search_service = SearchService(context)

        if search_type == SearchType.WEATHER:
            return await WeatherService.get_weather(search_query, search_service)
        elif search_type == SearchType.REDDIT:
            return await search_service.perform_search(SearchType.REDDIT, f"site:reddit.com {search_query}")
        elif search_type == SearchType.WIKIPEDIA:
            return await search_service.perform_search(SearchType.WIKIPEDIA, f"site:wikipedia.org {search_query}")
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
    def strip_search_id_tag(text: str) -> str:
        """Remove [search_id: N] tag from text."""
        return re.sub(Patterns.SEARCH_ID_TAG, '', text, flags=re.IGNORECASE).strip()

    @staticmethod
    async def extract_search_query(context: ChatContext, original_response: str) -> SearchResult:
        """
        Use specialized prompt to extract search query from user question.
        This is called when the model mentioned knowledge cutoff or when pre-flight triggers.
        """
        user_context = ChatService._format_user_context(context.user_memory)
        extraction_system_prompt = SEARCH_QUERY_EXTRACTION_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            user_context = user_context
        )

        extraction_messages = ChatService._build_messages(
            request=context.request,
            system_prompt=extraction_system_prompt,
            strip_search_ids=True
        )

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

        search_results, source_url, search_id = await ChatService.execute_search(context, search_type, search_query)

        return SearchResult(
            performed=True,
            search_type=search_type,
            search_query=search_query,
            search_results=search_results,
            source_url=source_url,
            search_id=search_id
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
        return ChatService._build_messages(
            request=request,
            system_prompt=system_prompt
        )

    @staticmethod
    async def detect_and_recall_from_cache(assistant_response: str) -> tuple[bool, int | None, SearchResult]:
        """Detect if LLM requested a recall and retrieve from cache."""
        from utils.cache import get_search_cache

        search_id = ChatService._parse_recall_command(assistant_response)

        if not search_id:
            return False, None, SearchResult(performed=False)

        app_logger.info(f"RECALL triggered for search ID: {search_id}")

        # Retrieve from cache
        cache = get_search_cache()
        cached_result = cache.get_by_id(search_id)

        if not cached_result:
            app_logger.warning(f"RECALL failed: search ID {search_id} not found in cache")
            return True, search_id, SearchResult(performed=False)

        search_results, source_url, metadata = cached_result

        return True, search_id, SearchResult(
            performed=True,
            search_type=metadata.get("search_type"),
            search_query=metadata.get("query"),
            search_results=search_results,
            source_url=source_url,
            search_id=search_id
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
    def _parse_recall_command(text: str) -> int | None:
        """Parse RECALL command from text and extract search ID."""
        recall_match = re.search(Patterns.RECALL, text, re.IGNORECASE)
        if recall_match:
            try:
                search_id = int(recall_match.group(1))
                return search_id
            except ValueError:
                return None
        return None

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

        # Fallback pattern for simple "SEARCH: <query>"
        fallback_match = re.search(Patterns.SEARCH_FALLBACK, text, re.IGNORECASE)
        if fallback_match:
            search_query = fallback_match.group(1).strip()
            return "NEEDS_QUERY_EXTRACTION", search_query

        return None, None

    @staticmethod
    def _build_messages(request: ChatRequest, system_prompt: str, additional_reserve: int = 0, validate: bool = True, strip_search_ids: bool = False) -> list:
        """Build messages list with system prompt, history, and user prompt.

        Args:
            request: Chat request containing prompt and history
            system_prompt: System prompt to use
            additional_reserve: Additional tokens to reserve for search results
            validate: Whether to validate context limits and raise on overflow
            strip_search_ids: If True, strip [search_id: N] tags from history content (for synthesis calls)

        Returns:
            List of messages ready for LLM
        """
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # Truncate history to fit within token budget
        if request.history:
            formatted_history = []
            for msg in request.history:
                content = msg.content

                if strip_search_ids:
                    content = ChatService.strip_search_id_tag(content)
                elif msg.search_id:
                    content = f"{content} [search_id: {msg.search_id}]"

                formatted_history.append({"role": msg.role, "content": content})
            app_logger.debug(f"Formatted history before truncation: {formatted_history}")

            truncated_history, messages_included = TokenManager.truncate_history_to_fit(
                system_prompt=system_prompt,
                user_memory=request.user_memory or "",
                current_prompt=request.prompt,
                history=formatted_history,
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
    def _prepare_search_response_messages(context: ChatContext, search_results: str, is_recall: bool = False) -> list:
        """Prepare messages with search results injected into system prompt.

        Args:
            context: ChatContext with request data
            search_results: Search results to include in prompt
            is_recall: If True, use RECALL-specific synthesis prompt
        """
        # Choose prompt based on whether this is a recall or new search
        prompt_template = RECALL_SYNTHESIS_PROMPT if is_recall else SEARCH_RESULT_SYSTEM_PROMPT

        search_system_prompt = prompt_template.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            user_context=ChatService._format_user_context(context.user_memory),
            search_results=search_results
        )

        # Calculate search result token cost
        search_result_tokens = TokenManager.estimate_tokens(search_results)
        app_logger.info(f"Search results size: {search_result_tokens} tokens")

        return ChatService._build_messages(
            request=context.request,
            system_prompt=search_system_prompt,
            additional_reserve=search_result_tokens,
            validate=False,
            strip_search_ids=True
        )

    @staticmethod
    async def _yield_search_and_response(search_result: SearchResult, context: ChatContext, is_recall: bool = False) -> AsyncIterator[FlowStep]:
        """
        Yield search action followed by stream response with search results.

        Args:
            search_result: SearchResult object with search data
            context: ChatContext with request data
            is_recall: If True, indicates this is a RECALL operation (not a new search)

        Yields:
            FlowStep for search action and stream response
        """
        # Yield search action (or recall action)
        if is_recall:
            yield FlowStep(
                action=FlowAction.RECALL,
                recall_id=search_result.search_id
            )
        else:
            yield FlowStep(
                action=FlowAction.SEARCH,
                search_type=search_result.search_type,
                search_query=search_result.search_query
            )

        # Prepare final messages with search results
        final_messages = ChatService._prepare_search_response_messages(
            context,
            search_result.search_results,
            is_recall=is_recall
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
        """Process LLM response and handle search/recall/cutoff detection.

        Args:
            context: ChatContext with request data
            call_num: LLM call number for logging

        Yields:
            FlowStep for various actions (search, recall, response, etc.)
        """
        response = await context.client.chat(
            model=context.request.model,
            messages=context.messages
        )
        assistant_response = response['message']['content']
        app_logger.debug(f"LLM Call #{call_num} response preview: {assistant_response[:100]}...")

        # Check for RECALL tag
        recall_detected, recall_id, recall_result = await ChatService.detect_and_recall_from_cache(assistant_response)

        if recall_detected:
            if recall_result.performed:
                async for step in ChatService._yield_search_and_response(recall_result, context, is_recall=True):
                    yield step
                return
            else:
                # RECALL failed - search ID not found
                app_logger.warning(f"RECALL {recall_id} failed, rerouting to new search")
                yield FlowStep(action=FlowAction.RECALL_FAILED)

                # Reroute to new search flow
                search_result = await ChatService.extract_search_query(context, assistant_response)
                async for step in ChatService._yield_search_and_response(search_result, context):
                    yield step
                return

        # Check for SEARCH tag
        search_type, search_query = ChatService._parse_search_command(assistant_response)

        if search_type:
            if search_type == "NEEDS_QUERY_EXTRACTION":
                # Simple "SEARCH: <query>" detected, reroute to search query extraction
                app_logger.info("Simple SEARCH tag detected, rerouting to search query extraction")
                search_result = await ChatService.extract_search_query(context, assistant_response)
                async for step in ChatService._yield_search_and_response(search_result, context):
                    yield step
                return
            else:
                # Typed search detected, perform it directly
                app_logger.info(f"Search triggered: {search_type} - '{search_query}'")
                search_results, source_url, search_id = await ChatService.execute_search(context, search_type, search_query)
                search_result = SearchResult(
                    performed=True,
                    search_type=search_type,
                    search_query=search_query,
                    search_results=search_results,
                    source_url=source_url,
                    search_id=search_id
                )
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

            if search_result.search_id:
                metadata["search_id"] = search_result.search_id

        return metadata