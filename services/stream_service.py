"""
Streaming service containing core streaming logic.
Handles real-time sanitization, cutoff detection, and SSE formatting.
"""
import json
from typing import AsyncIterator
from models.api_models import ChatRequest
from models.chat_models import ChatContext, SearchResult
from services.chat_service import ChatService
from utils.logger import app_logger
from utils.streaming_sanitizer import StreamingSanitizer
from config import Config


class StreamService:
    """Service for handling streaming chat operations."""

    @staticmethod
    def send_sse_event(event_type: str, data: dict) -> str:
        """Format data as Server-Sent Events (SSE) format."""
        return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"

    @staticmethod
    async def stream_with_realtime_sanitization_and_cutoff_detection(
        client,
        request: ChatRequest,
        messages: list,
        context: ChatContext,
        call_number: int,
        max_cutoff_retries: int = None,
        search_result: SearchResult = None,
        max_tag_retries: int = 3
    ) -> AsyncIterator[str]:
        """Stream LLM response with real-time sanitization, cutoff, SEARCH tags, and RECALL tag detection.
        
        Args:
            client: Ollama client
            request: Chat request
            messages: Messages to send to LLM
            context: Chat context
            call_number: LLM call number for logging
            max_cutoff_retries: Max retry attempts for cutoff detection (defaults to Config)
            search_result: SearchResult to include in metadata
            max_tag_retries: Max retry attempts for SEARCH/RECALL tag detection
            
        Yields:
            SSE events for status, tokens, and completion
        """
        if max_cutoff_retries is None:
            max_cutoff_retries = Config.MAX_KNOWLEDGE_CUTOFF_RETRIES
        
        if search_result is None:
            search_result = SearchResult(performed=False)
        
        for attempt in range(1, max_cutoff_retries + 2):
            cutoff_detected = False
            tag_detected = False
            skip_first_line_buffering = attempt > max_cutoff_retries
            
            if skip_first_line_buffering:
                app_logger.info(f"Max retries exceeded for Call #{call_number}, streaming immediately without buffering")
            
            # Initialize sanitizer
            sanitizer = StreamingSanitizer()
            
            # Start LLM stream
            llm_stream = client.chat(
                model=request.model,
                messages=messages,
                stream=True
            )
            
            first_line_buffer = ""
            first_line_complete = False
            token_count = 0
            full_response_for_metadata = ""
            tokens_buffered_count = 0
            tag_pending = False
            tokens_since_tag_detected = 0
            max_tokens_after_tag = 15
            tag_detected_at_token = None
            
            stream_iterator = None
            try:
                stream_iterator = await llm_stream
                async for chunk in stream_iterator:
                    token = chunk['message']['content']
                    token_count += 1
                    
                    if not first_line_complete and not skip_first_line_buffering:
                        tokens_buffered_count += 1
                        first_line_buffer += token

                        if tag_pending:
                            tokens_since_tag_detected += 1

                        # Check for query completion delimiters when tag is pending
                        query_complete = False
                        if tag_pending:
                            has_newline = '\n' in token
                            has_period = '.' in token
                            max_tokens_reached = tokens_since_tag_detected >= max_tokens_after_tag
                            
                            if has_newline or has_period or max_tokens_reached:
                                query_complete = True

                        has_enough_for_tag_check = len(first_line_buffer) >= 7
                        should_check_tags = has_enough_for_tag_check and not tag_pending

                        buffer_has_tag_prefix = False
                        if not tag_pending and len(first_line_buffer) >= 6:
                            for prefix in ["REDDIT:", "GOOGLE:", "WIKI:", "WIKIPEDIA:", "WEATHER:", "SEARCH:", "RECALL:"]:
                                if prefix in first_line_buffer.upper():
                                    buffer_has_tag_prefix = True
                                    break
                        
                        # Check at token milestones for cutoff
                        line_is_complete = '\n' in token or token_count >= 100
                        should_check_cutoff = (line_is_complete or len(first_line_buffer) >= 15) and not tag_pending
                        
                        if should_check_cutoff or should_check_tags or buffer_has_tag_prefix:
                            # Check for RECALL tag detection
                            if should_check_tags:
                                recall_detected, recall_id, recall_result = await ChatService.detect_and_recall_from_cache(first_line_buffer)
                                if recall_detected:
                                    app_logger.info(f"RECALL tag detected in first line: {recall_id}")
                                    tag_pending = True
                                    tag_detected = True
                                    tag_detected_at_token = token_count
                                    tokens_since_tag_detected = 0
                                
                                # Check for SEARCH tag detection
                                if not tag_pending:
                                    search_type, search_query = ChatService._parse_search_command(first_line_buffer)
                                    if search_type:
                                        app_logger.info(f"SEARCH tag detected in first line: {search_type}")
                                        tag_pending = True
                                        tag_detected = True
                                        tag_detected_at_token = token_count
                                        tokens_since_tag_detected = 0
                            
                            # Check for cutoff detection
                            if should_check_cutoff:
                                if ChatService.detect_knowledge_cutoff(first_line_buffer):
                                    app_logger.warning(f"Knowledge cutoff detected in first line (attempt {attempt}/{max_cutoff_retries})")
                                    cutoff_detected = True
                                    tag_pending = True
                                    tag_detected_at_token = token_count
                                    tokens_since_tag_detected = 0

                        if query_complete:
                            break

                        if line_is_complete and not tag_pending and not tag_detected:
                            if buffer_has_tag_prefix:
                                pass
                            else:
                                first_line_complete = True
                                app_logger.info(f"First line complete ({tokens_buffered_count} tokens buffered), no tags/cutoff detected")
                                
                                # Add verified clean buffer to full response
                                full_response_for_metadata += first_line_buffer
                                
                                # First line verified clean - output directly
                                yield StreamService.send_sse_event("token", {"content": first_line_buffer})
                                
                                continue
                    else:
                        if not tag_pending:
                            full_response_for_metadata += token

                    if first_line_complete or skip_first_line_buffering:
                        sanitized = sanitizer.process_token(token)
                        if sanitized:
                            # Add sanitized content to full response for metadata
                            full_response_for_metadata += sanitized
                            yield StreamService.send_sse_event("token", {"content": sanitized})
            
            finally:
                # Stream ended - do final tag check on any accumulated response
                if not skip_first_line_buffering and not tag_detected and full_response_for_metadata:
                    
                    # Final check for tags in the full response
                    recall_detected, recall_id, recall_result = await ChatService.detect_and_recall_from_cache(full_response_for_metadata)
                    if recall_detected:
                        app_logger.info(f"RECALL tag detected in final check: {recall_id}")
                        tag_detected = True
                        tag_detected_at_token = token_count
                    
                    if not tag_detected:
                        search_type, search_query = ChatService._parse_search_command(full_response_for_metadata)
                        if search_type:
                            app_logger.info(f"SEARCH tag detected in final check: {search_type}")
                            tag_detected = True
                            tag_detected_at_token = token_count
                            full_response_for_metadata = ""

                if first_line_buffer and not first_line_complete and not skip_first_line_buffering and not tag_detected and not cutoff_detected:
                    app_logger.info(f"Stream ended while buffering ({tokens_buffered_count} tokens), outputting verified clean buffer")
                    # Add buffered content to full response for metadata
                    full_response_for_metadata += first_line_buffer
                    yield StreamService.send_sse_event("token", {"content": first_line_buffer})
                elif tag_detected and tag_detected_at_token:
                    app_logger.info(f"Tag detected at token {tag_detected_at_token}/{tokens_buffered_count}, buffer not output (will process tag)")
                
                if stream_iterator is not None:
                    try:
                        await stream_iterator.aclose()
                        if cutoff_detected:
                            app_logger.info(f"Stream closed after knowledge cutoff detection (attempt {attempt})")
                        elif tag_detected:
                            app_logger.info(f"Stream closed after tag detection (SEARCH/RECALL) at token {tag_detected_at_token}")
                    except Exception as e:
                        app_logger.warning(f"Error closing stream: {e}")
            
            # Handle tag detection (SEARCH/RECALL)
            if tag_detected:
                # check RECALL tag
                recall_detected, recall_id, recall_result = await ChatService.detect_and_recall_from_cache(first_line_buffer)
                
                if recall_detected:
                    if recall_result.performed:
                        yield StreamService.send_sse_event("status", {
                            "stage": "recalling",
                            "message": "Let me look at it..."
                        })
                        
                        search_result = recall_result
                        is_recall = True
                    else:
                        # Invalid recall ID - notify and extract new query
                        yield StreamService.send_sse_event("status", {
                            "stage": "recall_failed",
                            "message": "I couldn't find the resources, It may have expired. Let me do a new search!"
                        })
                        
                        search_result = await ChatService.extract_search_query(context, first_line_buffer)
                        is_recall = False
                else:
                    search_type, search_query = ChatService._parse_search_command(first_line_buffer)
                    
                    if search_type:
                        if search_type == "NEEDS_QUERY_EXTRACTION":
                            # Simple "SEARCH:" detected, need query extraction
                            app_logger.info("Simple SEARCH tag detected in stream, extracting query")
                            search_result = await ChatService.extract_search_query(context, first_line_buffer)
                        else:
                            # Typed search detected (e.g., GOOGLE:, WIKIPEDIA:)
                            app_logger.info(f"Typed SEARCH tag detected in stream: {search_type} - '{search_query}'")
                            search_result = await ChatService.validate_and_execute_search(context, search_type, search_query)
                        
                        is_recall = False
                    else:
                        # Knowledge cutoff detected
                        yield StreamService.send_sse_event("status", {
                            "stage": "rerouting",
                            "message": f"Detecting knowledge limitation, searching for current information..."
                        })
                        
                        search_result = await ChatService.extract_search_query(context, first_line_buffer)
                        is_recall = False

                yield StreamService.send_sse_event("status", {
                    "stage": "searching",
                    "message": f"Searching {search_result.search_type} for: {search_result.search_query}"
                })
                
                # Prepare new messages with search results
                messages = ChatService._prepare_search_response_messages(
                    context,
                    search_result.search_results,
                    is_recall=is_recall
                )

                call_number = context.next_call_number() if is_recall else context.call_count
                yield StreamService.send_sse_event("status", {"stage": "generating"})

                async for event in StreamService.stream_with_realtime_sanitization_and_cutoff_detection(
                    client=client,
                    request=request,
                    messages=messages,
                    context=context,
                    call_number=call_number,
                    max_cutoff_retries=max_cutoff_retries,
                    search_result=search_result,
                    max_tag_retries=max_tag_retries
                ):
                    yield event
                return

            if cutoff_detected and attempt <= max_cutoff_retries:
                # Knowledge cutoff detected - prepare retry with search
                yield StreamService.send_sse_event("status", {
                    "stage": "rerouting",
                    "message": f"Detecting knowledge limitation, searching for current information..."
                })
                
                search_result = await ChatService.extract_search_query(context, first_line_buffer)
                
                # Emit search status
                yield StreamService.send_sse_event("status", {
                    "stage": "searching",
                    "message": f"Searching {search_result.search_type} for: {search_result.search_query}"
                })
                
                # Prepare new messages with search results
                messages = ChatService._prepare_search_response_messages(
                    context,
                    search_result.search_results,
                    is_recall=False
                )

                call_number = context.call_count
                yield StreamService.send_sse_event("status", {"stage": "generating"})
                
                continue
            
            # Successfully streamed or max retries exceeded
            remaining = sanitizer.flush()
            if remaining:
                yield StreamService.send_sse_event("token", {"content": remaining})

            metadata = ChatService.build_response_metadata(request, messages, search_result)
            metadata["full_response"] = full_response_for_metadata.rstrip()
            metadata["cutoff_retries"] = attempt - 1 if cutoff_detected else 0
            yield StreamService.send_sse_event("done", metadata)
            return
