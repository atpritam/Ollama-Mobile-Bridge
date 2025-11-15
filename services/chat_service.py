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


class ChatService:
    """Service for handling chat logic."""

    SEARCH_TYPE_MAPPING = {
        'WEATHER': SearchType.WEATHER,
        'GOOGLE': SearchType.GOOGLE,
        'REDDIT': SearchType.REDDIT,
        'WIKI': SearchType.WIKIPEDIA
    }

    @staticmethod
    def direct_search_detection(user_query: str) -> tuple[bool, str, str]:
        """Detect explicit search intent"""
        query_lower = user_query.lower()
        # Reddit keywords
        if re.search(r"reddit|people think|opinions|people saying about|opinion|reviews", query_lower):
            return True, SearchType.REDDIT, user_query

        # Wikipedia keywords
        if re.search(r"wikipedia|wiki", query_lower):
            return True, SearchType.WIKIPEDIA, user_query

        # Google search for recency
        if re.search(r"recent|latest|2025|2026|this year", query_lower):
            return True, SearchType.GOOGLE, user_query

        return False, "", ""

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

        search_match = re.search(Patterns.SEARCH_WITH_TYPE, extraction_response, re.IGNORECASE)

        if not search_match:
            fallback_match = re.search(Patterns.SEARCH_FALLBACK, extraction_response, re.IGNORECASE)
            if fallback_match:
                search_type = SearchType.GOOGLE
                search_query = fallback_match.group(1).strip()
                app_logger.info(f"Extraction fallback: google - '{search_query}'")
            else:
                app_logger.warning(f"Extraction failed, using original query")
                search_type = SearchType.GOOGLE
                search_query = context.prompt
        else:
            search_type_raw = search_match.group(1).upper()
            search_query = search_match.group(2).strip()
            search_query = search_query.strip('"').strip("'")

            search_type = ChatService.parse_search_type(search_type_raw)
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

        if Config.is_small_model(request.model):
            return SIMPLE_SYSTEM_PROMPT.format(
                current_date=datetime.now().strftime("%Y-%m-%d")
            )

        return DEFAULT_SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

    @staticmethod
    def prepare_messages(request: ChatRequest, system_prompt: str) -> list:
        """
        Prepare a messages list for LLM with system prompt and history.

        Args:
            request: Chat request containing prompt and history
            system_prompt: System prompt to use

        Returns:
            List of messages for LLM
        """
        messages = []

        # System prompt
        messages.append({"role": "system", "content": system_prompt})

        # History (last N messages)
        if request.history:
            limited_history = request.history[-Config.MAX_HISTORY_MESSAGES:]
            messages.extend([{"role": msg.role, "content": msg.content} for msg in limited_history])

        # Current user prompt
        messages.append({"role": "user", "content": request.prompt})

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
        search_match = re.search(Patterns.SEARCH_WITH_TYPE, assistant_response, re.IGNORECASE)

        if not search_match:
            fallback_match = re.search(Patterns.SEARCH_FALLBACK, assistant_response, re.IGNORECASE)
            if fallback_match:
                search_type = SearchType.GOOGLE
                search_query = fallback_match.group(1).strip()
                app_logger.info(f"Search triggered (default to google): '{search_query}'")
            else:
                return SearchResult(performed=False)
        else:
            search_type_raw = search_match.group(1).upper()
            search_query = search_match.group(2).strip()
            search_query = search_query.strip('"').strip("'")

            search_type = ChatService.parse_search_type(search_type_raw)
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
    async def orchestrate_chat_flow(context: ChatContext) -> AsyncIterator[FlowStep]:
        """Core flow orchestrator that yields decision points."""
        is_small = Config.is_small_model(context.model_name)

        # DIRECT SEARCH DETECTION
        # ============================================================
        has_direct_intent, search_type, search_query = ChatService.direct_search_detection(context.prompt)

        if has_direct_intent:
            app_logger.info(f"Direct search: {search_type} - '{search_query}'")

            # Yield search action
            yield FlowStep(
                action=FlowAction.SEARCH,
                search_type=search_type,
                search_query=search_query
            )

            # Execute search
            search_results, source_url = await ChatService.execute_search(context, search_type, search_query)

            search_result = SearchResult(
                performed=True,
                search_type=search_type,
                search_query=search_query,
                search_results=search_results,
                source_url=source_url
            )

            # Prepare final messages with search results
            final_messages = ChatService.prepare_messages(
                context.request,
                SEARCH_RESULT_SYSTEM_PROMPT.format(
                    current_date=datetime.now().strftime("%Y-%m-%d"),
                    search_results=search_results
                )
            )

            # Yield stream response action
            yield FlowStep(
                action=FlowAction.STREAM_RESPONSE,
                messages=final_messages,
                search_result=search_result
            )
            return

        if is_small:
            # SMALL MODEL - Scenario A: Pre-flight detects a recency pattern
            if ChatService.preflight_search_check(context.prompt):
                # Call #1: Generate proper search query using specialized prompt
                search_result = await ChatService.extract_search_query(context, "")

                # Yield search action
                yield FlowStep(
                    action=FlowAction.SEARCH,
                    search_type=search_result.search_type,
                    search_query=search_result.search_query
                )

                # Prepare final messages with search results
                final_messages = ChatService.prepare_messages(
                    context.request,
                    SEARCH_RESULT_SYSTEM_PROMPT.format(
                        current_date=datetime.now().strftime("%Y-%m-%d"),
                        search_results=search_result.search_results
                    )
                )

                # Yield stream response action
                yield FlowStep(
                    action=FlowAction.STREAM_RESPONSE,
                    messages=final_messages,
                    search_result=search_result
                )
                return

            # SMALL MODEL - Scenario B: Pre-flight passes, initial call with simple prompt
            call_num = context.next_call_number()
            app_logger.info(f"LLM Call #{call_num}: Pre-flight passed, attempting with simple prompt")
            response = await context.client.chat(
                model=context.request.model,
                messages=context.messages
            )
            assistant_response = response['message']['content']
            app_logger.debug(f"LLM Call #{call_num} response preview: {assistant_response[:100]}...")

            # Check for SEARCH tag in response
            search_result = await ChatService.detect_and_perform_search(assistant_response, context)

            if search_result.performed:
                # SEARCH tag detected, perform search and synthesize
                app_logger.info("SEARCH tag detected in Call #1")

                # Yield search action
                yield FlowStep(
                    action=FlowAction.SEARCH,
                    search_type=search_result.search_type,
                    search_query=search_result.search_query
                )

                # Prepare final messages with search results
                final_messages = ChatService.prepare_messages(
                    context.request,
                    SEARCH_RESULT_SYSTEM_PROMPT.format(
                        current_date=datetime.now().strftime("%Y-%m-%d"),
                        search_results=search_result.search_results
                    )
                )

                # Yield stream response action
                yield FlowStep(
                    action=FlowAction.STREAM_RESPONSE,
                    messages=final_messages,
                    search_result=search_result
                )
                return

            # Check for knowledge cutoff pattern
            if ChatService.detect_knowledge_cutoff(assistant_response):
                app_logger.info("Cutoff detected, triggering Call #2 for query extraction")

                # Call #2: Extract search query
                search_result = await ChatService.extract_search_query(context, assistant_response)

                # Yield search action
                yield FlowStep(
                    action=FlowAction.SEARCH,
                    search_type=search_result.search_type,
                    search_query=search_result.search_query
                )

                # Prepare final messages with search results
                final_messages = ChatService.prepare_messages(
                    context.request,
                    SEARCH_RESULT_SYSTEM_PROMPT.format(
                        current_date=datetime.now().strftime("%Y-%m-%d"),
                        search_results=search_result.search_results
                    )
                )

                # Yield stream response action
                yield FlowStep(
                    action=FlowAction.STREAM_RESPONSE,
                    messages=final_messages,
                    search_result=search_result
                )
                return

            # No search needed, return Call #1 response
            clean_response = ChatService.clean_response(assistant_response)
            yield FlowStep(
                action=FlowAction.RETURN_RESPONSE,
                response=clean_response,
                messages=context.messages,  # Pass messages for streaming
                search_result=SearchResult(performed=False),
                call_number=call_num
            )
            return

        else:
            # LARGE MODEL: Skip pre-flight, directly initial call with complex prompt
            call_num = context.next_call_number()
            app_logger.info(f"LLM Call #{call_num}: Large model with complex prompt")
            response = await context.client.chat(
                model=context.request.model,
                messages=context.messages
            )
            assistant_response = response['message']['content']
            app_logger.debug(f"LLM Call #{call_num} response preview: {assistant_response[:100]}...")

            # Check for SEARCH tag
            search_result = await ChatService.detect_and_perform_search(assistant_response, context)

            if search_result.performed:
                # SEARCH tag detected, perform search and synthesize
                app_logger.info("SEARCH tag detected in Call #1")

                # Yield search action
                yield FlowStep(
                    action=FlowAction.SEARCH,
                    search_type=search_result.search_type,
                    search_query=search_result.search_query
                )

                # Prepare final messages with search results
                final_messages = ChatService.prepare_messages(
                    context.request,
                    SEARCH_RESULT_SYSTEM_PROMPT.format(
                        current_date=datetime.now().strftime("%Y-%m-%d"),
                        search_results=search_result.search_results
                    )
                )

                # Yield stream response action
                yield FlowStep(
                    action=FlowAction.STREAM_RESPONSE,
                    messages=final_messages,
                    search_result=search_result
                )
                return

            # Check for knowledge cutoff pattern
            if ChatService.detect_knowledge_cutoff(assistant_response):
                app_logger.info("Cutoff detected, triggering Call #2 for query extraction")

                # Call #2: Extract search query
                search_result = await ChatService.extract_search_query(context, assistant_response)

                # Yield search action
                yield FlowStep(
                    action=FlowAction.SEARCH,
                    search_type=search_result.search_type,
                    search_query=search_result.search_query
                )

                # Prepare final messages with search results
                final_messages = ChatService.prepare_messages(
                    context.request,
                    SEARCH_RESULT_SYSTEM_PROMPT.format(
                        current_date=datetime.now().strftime("%Y-%m-%d"),
                        search_results=search_result.search_results
                    )
                )

                # Yield stream response action
                yield FlowStep(
                    action=FlowAction.STREAM_RESPONSE,
                    messages=final_messages,
                    search_result=search_result
                )
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
            return

    @staticmethod
    def build_response_metadata(request: ChatRequest,messages: list,search_result: SearchResult) -> dict:
        """Build response metadata dictionary."""
        metadata = {
            "model": request.model,
            "context_messages_count": len(messages) - 1,
            "search_performed": search_result.performed
        }

        if search_result.performed:
            metadata["search_type"] = search_result.search_type
            metadata["search_query"] = search_result.search_query

            source_domain = ChatService.extract_domain(search_result.source_url)
            if source_domain:
                metadata["source"] = source_domain

        return metadata