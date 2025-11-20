"""
Route handlers for chat operations.
Handles both standard and streaming chat endpoints.
"""
import json
from typing import AsyncIterator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import ollama
from models.api_models import ChatRequest
from models.chat_models import ChatContext, SearchResult, FlowAction
from services.chat_service import ChatService
from utils.logger import app_logger

router = APIRouter()

def send_sse_event(event_type: str, data: dict) -> str:
    """Format data as Server-Sent Events (SSE) format."""
    return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"

def send_context_limit_error(e) -> dict:
    """Send token usage context limit error."""
    return {
        "error": "context_limit_exceeded",
        "message": str(e),
        "suggestions": [
            "Start a new chat",
            "Try a model with a larger context window",
            "Shorten your current prompt or user memory"
        ]
    }


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation history and automatic web search.
    Supports up to the last 30 messages as context.
    """
    try:
        client = ollama.AsyncClient()
        system_prompt = ChatService.get_system_prompt(request)
        messages = ChatService.prepare_messages(request, system_prompt)

        context = ChatContext(
            request=request,
            client=client,
            messages=messages,
            system_prompt=system_prompt
        )

        final_response = None
        search_result = SearchResult(performed=False)

        # Process flow steps
        async for step in ChatService.orchestrate_chat_flow(context):
            if step.action == FlowAction.RETURN_RESPONSE:
                # Direct response without streaming
                final_response = step.response
                search_result = step.search_result
                break

            elif step.action == FlowAction.STREAM_RESPONSE:
                # Need to synthesize with search results
                call_num = context.next_call_number()
                app_logger.info(f"LLM Call #{call_num}: Synthesizing final response with search results")
                response = await client.chat(
                    model=request.model,
                    messages=step.messages
                )
                final_response = response['message']['content']
                app_logger.info(f"LLM Call #{call_num} completed: Generated {len(final_response)} characters")
                search_result = step.search_result
                break

        # Build and return response
        response_data = ChatService.build_response_metadata(request, messages, search_result)
        final_response = ChatService.strip_search_id_tag(final_response)
        response_data["response"] = final_response

        return response_data

    except ValueError as e:
        app_logger.error(f"Context limit error: {str(e)}")
        return send_context_limit_error(e)
    except ollama.ResponseError as e:
        app_logger.error(f"Ollama error: {e.error}")
        return {"error": "ollama_error", "message": e.error}
    except Exception as e:
        app_logger.error(f"Chat error: {str(e)}")
        return {"error": str(e)}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with real-time status updates.
    Returns Server-Sent Events (SSE) stream with:
    - status updates (thinking, searching, scraping, generating)
    - token-by-token response streaming
    - final response metadata
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

            # Process flow steps
            async for step in ChatService.orchestrate_chat_flow(context):
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

                    app_logger.info(f"LLM Call #{step.call_number}: Streaming response")

                    full_response = ""
                    async for chunk in await client.chat(
                        model=request.model,
                        messages=step.messages,
                        stream=True
                    ):
                        token = chunk['message']['content']
                        full_response += token
                        yield send_sse_event("token", {"content": token})

                    app_logger.info(f"LLM Call #{step.call_number} completed: Streamed {len(full_response)} characters")
                    full_response = ChatService.strip_search_id_tag(full_response)

                    metadata = ChatService.build_response_metadata(request, messages, step.search_result)
                    metadata["full_response"] = full_response
                    yield send_sse_event("done", metadata)
                    return

                elif step.action == FlowAction.STREAM_RESPONSE:
                    if step.search_result.performed and step.search_result.source_url:
                        source_domain = ChatService.extract_domain(step.search_result.source_url)
                        if source_domain:
                            yield send_sse_event("status", {
                                "stage": "reading_content",
                                "message": f"Reading content from {source_domain}"
                            })

                    # Stream final response
                    yield send_sse_event("status", {"stage": "generating"})

                    call_num = context.next_call_number()
                    app_logger.info(f"LLM Call #{call_num}: Synthesizing final response with search results (streaming)")

                    full_response = ""
                    async for chunk in await client.chat(
                        model=request.model,
                        messages=step.messages,
                        stream=True
                    ):
                        token = chunk['message']['content']
                        full_response += token
                        yield send_sse_event("token", {"content": token})

                    app_logger.info(f"LLM Call #{call_num} completed: Streamed {len(full_response)} characters")
                    full_response = ChatService.strip_search_id_tag(full_response)

                    # final metadata
                    metadata = ChatService.build_response_metadata(request, messages, step.search_result)
                    metadata["full_response"] = full_response
                    yield send_sse_event("done", metadata)
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