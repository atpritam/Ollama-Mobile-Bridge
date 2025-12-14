"""
Route handlers for streaming chat operations.
Handles the /chat/stream endpoint with real-time status updates.
"""
import json
from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import ollama
from models.api_models import ChatRequest
from models.chat_models import ChatContext, FlowAction, SearchResult
from services.chat_service import ChatService
from utils.logger import app_logger
from utils.streaming_sanitizer import StreamingSanitizer
from routes.chat import send_context_limit_error
from config import Config

router = APIRouter()

def send_sse_event(event_type: str, data: dict) -> str:
    """Format data as Server-Sent Events (SSE) format."""
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


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
    """Stream LLM response with real-time sanitization, cutoff, SEARCH tag, and RECALL tag detection.
    
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
        tag_detected = False  # Separate flag for SEARCH/RECALL tags
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
        
        stream_iterator = None
        try:
            stream_iterator = await llm_stream
            async for chunk in stream_iterator:
                token = chunk['message']['content']
                token_count += 1
                full_response_for_metadata += token
                
                if not first_line_complete and not skip_first_line_buffering:
                    tokens_buffered_count += 1
                    first_line_buffer += token

                    line_is_complete = '\n' in token or token_count >= 80

                    has_enough_for_tag_check = len(first_line_buffer) >= 7
                    should_check_tags = has_enough_for_tag_check or line_is_complete
                    should_check_cutoff = (token_count % 3 == 0) or line_is_complete
                    
                    if should_check_cutoff or should_check_tags:
                        # Check for RECALL tag detection
                        if should_check_tags:
                            recall_detected, recall_id, recall_result = await ChatService.detect_and_recall_from_cache(first_line_buffer)
                            if recall_detected:
                                app_logger.info(f"RECALL tag detected in first line: {recall_id}")
                                tag_detected = True
                                break
                            
                            # Check for SEARCH tag detection
                            search_type, search_query = ChatService._parse_search_command(first_line_buffer)
                            if search_type:
                                app_logger.info(f"SEARCH tag detected in first line: {search_type}")
                                tag_detected = True
                                break
                        
                        # Check for cutoff detection
                        if should_check_cutoff and ChatService.detect_knowledge_cutoff(first_line_buffer):
                            app_logger.warning(f"Knowledge cutoff detected in first line (attempt {attempt}/{max_cutoff_retries})")
                            cutoff_detected = True
                            break
                    
                    if line_is_complete:
                        first_line_complete = True
                        app_logger.info(f"First line complete ({tokens_buffered_count} tokens buffered), no tags/cutoff detected")
                        
                        # First line verified clean - output directly without char-by-char processing
                        yield send_sse_event("token", {"content": first_line_buffer})
                        
                        continue

                if first_line_complete or skip_first_line_buffering:
                    sanitized = sanitizer.process_token(token)
                    if sanitized:
                        yield send_sse_event("token", {"content": sanitized})
        
        finally:
            if first_line_buffer and not first_line_complete and not skip_first_line_buffering and not tag_detected and not cutoff_detected:
                app_logger.info(f"Stream ended while buffering ({tokens_buffered_count} tokens), outputting verified clean buffer")
                yield send_sse_event("token", {"content": first_line_buffer})
            
            if stream_iterator is not None:
                try:
                    await stream_iterator.aclose()
                    if cutoff_detected:
                        app_logger.info(f"Stream closed after knowledge cutoff detection (attempt {attempt})")
                    elif tag_detected:
                        app_logger.info(f"Stream closed after tag detection (SEARCH/RECALL)")
                except Exception as e:
                    app_logger.warning(f"Error closing stream: {e}")
        
        # Handle tag detection (SEARCH/RECALL)
        if tag_detected:
            # check RECALL tag
            recall_detected, recall_id, recall_result = await ChatService.detect_and_recall_from_cache(first_line_buffer)
            
            if recall_detected:
                if recall_result.performed:
                    # Valid recall - use cached search results
                    yield send_sse_event("status", {
                        "stage": "recalling",
                        "message": "Let me look at it..."
                    })
                    
                    search_result = recall_result
                    is_recall = True
                else:
                    # Invalid recall ID - notify and extract new query
                    yield send_sse_event("status", {
                        "stage": "recall_failed",
                        "message": "I couldn't find the resources, It may have expired. Let me do a new search!"
                    })
                    
                    search_result = await ChatService.extract_search_query(context, first_line_buffer)
                    is_recall = False
            else:
                # Check if this was a SEARCH tag
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
                    yield send_sse_event("status", {
                        "stage": "rerouting",
                        "message": f"Detecting knowledge limitation, searching for current information... (attempt {attempt}/{max_cutoff_retries})"
                    })
                    
                    search_result = await ChatService.extract_search_query(context, first_line_buffer)
                    is_recall = False

            yield send_sse_event("status", {
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
            yield send_sse_event("status", {"stage": "generating"})
            
            # Recursively call with the new messages and reset attempt counter
            # This is a new LLM call, not a retry
            async for event in stream_with_realtime_sanitization_and_cutoff_detection(
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
        
        # Check if we need to retry (knowledge cutoff detected)
        if cutoff_detected and attempt <= max_cutoff_retries:
            # Knowledge cutoff detected - prepare retry with search
            yield send_sse_event("status", {
                "stage": "rerouting",
                "message": f"Detecting knowledge limitation, searching for current information... (attempt {attempt}/{max_cutoff_retries})"
            })
            
            search_result = await ChatService.extract_search_query(context, first_line_buffer)
            
            # Emit search status
            yield send_sse_event("status", {
                "stage": "searching",
                "message": f"Searching {search_result.search_type} for: {search_result.search_query}"
            })
            
            # Prepare new messages with search results
            messages = ChatService._prepare_search_response_messages(
                context,
                search_result.search_results,
                is_recall=False
            )
            
            # Get current call number (already incremented by extract_search_query)
            call_number = context.call_count
            yield send_sse_event("status", {"stage": "generating"})
            
            continue  # Retry with incremented attempt
        
        # Successfully streamed or max retries exceeded
        remaining = sanitizer.flush()
        if remaining:
            yield send_sse_event("token", {"content": remaining})
        
        # Send completion metadata
        metadata = ChatService.build_response_metadata(request, messages, search_result)
        metadata["full_response"] = full_response_for_metadata.rstrip()
        metadata["cutoff_retries"] = attempt - 1 if cutoff_detected else 0
        yield send_sse_event("done", metadata)
        return


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with real-time status updates and sanitization.
    """

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield send_sse_event("status", {"stage": "initializing"})

            client = ollama.AsyncClient()
            system_prompt = ChatService.get_system_prompt(request)
            messages = ChatService.prepare_messages(request, system_prompt)

            yield send_sse_event("status", {"stage": "thinking"})

            context = ChatContext(
                request=request,
                client=client,
                messages=messages,
                system_prompt=system_prompt
            )

            # Process flow steps (streaming mode)
            async for step in ChatService.orchestrate_chat_flow(context, is_streaming=True):
                if step.action == FlowAction.RECALL:
                    # Emit recall status
                    yield send_sse_event("status", {
                        "stage": "recalling",
                        "message": "Let me look at it..."
                    })

                elif step.action == FlowAction.RECALL_FAILED:
                    yield send_sse_event("status", {
                        "stage": "recall_failed",
                        "message": "I couldn't find the resources, It may have expired. Let me do a new search!"
                    })

                elif step.action == FlowAction.SEARCH:
                    # Emit search status
                    yield send_sse_event("status", {
                        "stage": f"searching",
                        "message": f"Searching {step.search_type} for: {step.search_query}",
                    })

                elif step.action == FlowAction.RETURN_RESPONSE:
                    yield send_sse_event("status", {"stage": "generating"})
                    
                    # Stream with real-time sanitization and cutoff detection
                    async for event in stream_with_realtime_sanitization_and_cutoff_detection(
                        client=client,
                        request=request,
                        messages=step.messages,
                        context=context,
                        call_number=step.call_number,
                        search_result=step.search_result
                    ):
                        yield event
                    return

                elif step.action == FlowAction.STREAM_RESPONSE:
                    if step.search_result.performed and step.search_result.source_url:
                        source_domain = ChatService.extract_domain(step.search_result.source_url)
                        if source_domain:
                            yield send_sse_event("status", {
                                "stage": "reading_content",
                                "message": f"Reading content from {source_domain}"
                            })

                    # Stream final response with real-time sanitization
                    yield send_sse_event("status", {"stage": "generating"})
                    
                    # Use call number from orchestrator step (already incremented)
                    call_num = step.call_number
                    
                    async for event in stream_with_realtime_sanitization_and_cutoff_detection(
                        client=client,
                        request=request,
                        messages=step.messages,
                        context=context,
                        call_number=call_num,
                        search_result=step.search_result
                    ):
                        yield event
                    return

        except ValueError as e:
            app_logger.error(f"Context limit error: {str(e)}")
            yield send_sse_event("error", send_context_limit_error(e))
        except ollama.ResponseError as e:
            app_logger.error(f"Ollama error: {e.error}")
            yield send_sse_event("error", {"type": "ollama_error", "message": e.error})
        except Exception as e:
            app_logger.error(f"Streaming chat error: {str(e)}")
            yield send_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
