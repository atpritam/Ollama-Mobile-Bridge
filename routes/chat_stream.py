"""
Route handlers for streaming chat operations.
Handles the /chat/stream endpoint with real-time status updates.
"""
from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import ollama
from models.api_models import ChatRequest
from models.chat_models import ChatContext, FlowAction
from services.chat_service import ChatService
from services.stream_service import StreamService
from utils.logger import app_logger
from routes.chat import send_context_limit_error

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with real-time status updates and sanitization.
    """

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield StreamService.send_sse_event("status", {"stage": "initializing"})

            client = ollama.AsyncClient()
            system_prompt = ChatService.get_system_prompt(request)
            messages = ChatService.prepare_messages(request, system_prompt)

            yield StreamService.send_sse_event("status", {"stage": "thinking"})

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
                    yield StreamService.send_sse_event("status", {
                        "stage": "recalling",
                        "message": "Let me look at it..."
                    })

                elif step.action == FlowAction.RECALL_FAILED:
                    yield StreamService.send_sse_event("status", {
                        "stage": "recall_failed",
                        "message": "I couldn't find the resources, It may have expired. Let me do a new search!"
                    })

                elif step.action == FlowAction.SEARCH:
                    # Emit search status
                    yield StreamService.send_sse_event("status", {
                        "stage": f"searching",
                        "message": f"Searching {step.search_type} for: {step.search_query}",
                    })

                elif step.action == FlowAction.RETURN_RESPONSE:
                    yield StreamService.send_sse_event("status", {"stage": "generating"})
                    
                    # Stream with real-time sanitization and cutoff detection
                    async for event in StreamService.stream_with_realtime_sanitization_and_cutoff_detection(
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
                            yield StreamService.send_sse_event("status", {
                                "stage": "reading_content",
                                "message": f"Reading content from {source_domain}"
                            })

                    # Stream final response with real-time sanitization
                    yield StreamService.send_sse_event("status", {"stage": "generating"})
                    call_num = step.call_number
                    
                    async for event in StreamService.stream_with_realtime_sanitization_and_cutoff_detection(
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
            yield StreamService.send_sse_event("error", send_context_limit_error(e))
        except ollama.ResponseError as e:
            app_logger.error(f"Ollama error: {e.error}")
            yield StreamService.send_sse_event("error", {"type": "ollama_error", "message": e.error})
        except Exception as e:
            app_logger.error(f"Streaming chat error: {str(e)}")
            yield StreamService.send_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
