"""
Debug route to see what the phone/client is sending
"""
from fastapi import APIRouter, Request
from utils.logger import app_logger

router = APIRouter()

@router.post("/chat/debug")
async def chat_stream_debug(request: Request):
    """Debug endpoint to see raw request body"""
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
        app_logger.info(f"Raw request body: {body_str}")
        
        import json
        parsed = json.loads(body_str)
        app_logger.info(f"Parsed JSON: {parsed}")
        
        return {"received": parsed, "status": "ok"}
    except Exception as e:
        app_logger.error(f"Debug error: {e}")
        return {"error": str(e)}
