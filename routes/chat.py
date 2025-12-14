"""
Route handlers for standard chat operations.
Handles the /chat endpoint (non-streaming).
"""
from fastapi import APIRouter
import ollama
from models.api_models import ChatRequest
from models.chat_models import ChatContext, SearchResult, FlowAction
from services.chat_service import ChatService
from utils.logger import app_logger

router = APIRouter()

async def handle_empty_response_reroute(context: ChatContext, client) -> tuple[str, SearchResult, list]:
    """Handle empty sanitized response by rerouting to simple prompt.
    
    Returns:
        Tuple of (final_response, search_result, messages)
    """
    app_logger.warning("Response empty after sanitization, rerouting to simple prompt")
    async for step in ChatService._handle_empty_sanitized_response(context):
        if step.action == FlowAction.RETURN_RESPONSE:
            final_response = ChatService.sanitize_final_response(step.response)
            return final_response, step.search_result, step.messages
        elif step.action == FlowAction.STREAM_RESPONSE:
            # Synthesize response
            call_num = context.next_call_number()
            app_logger.info(f"LLM Call #{call_num}: Synthesizing response after empty sanitization")
            response = await client.chat(
                model=context.request.model,
                messages=step.messages
            )
            final_response = ChatService.sanitize_final_response(response['message']['content'])
            return final_response, step.search_result, step.messages
    return "", SearchResult(performed=False), context.messages

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

        # Sanitize final response
        final_response = ChatService.sanitize_final_response(final_response)
        
        # Check if sanitized response is empty - reroute if needed
        if not final_response or final_response.strip() == "":
            final_response, search_result, messages = await handle_empty_response_reroute(context, client)
        
        # Build and return response
        response_data = ChatService.build_response_metadata(request, messages, search_result)
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