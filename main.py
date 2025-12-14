"""
Ollama Mobile Bridge - FastAPI application for chatting with local LLMs.
Featuring agentic System with real-time web access, autonomous search routing, cache retrieval and advanced context management.
"""
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import Config
from routes import chat, chat_stream, models_route, chat_debug
from auth import APIKeyMiddleware
from utils.http_client import HTTPClientManager
from utils.logger import app_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    yield
    await HTTPClientManager.close_all()

app = FastAPI(title=Config.APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with user-friendly messages"""
    errors = exc.errors()
    app_logger.error(f"Validation error for {request.url}")
    app_logger.error(f"Errors: {errors}")

    try:
       await request.body()
    except:
        pass

    if errors:
        first_error = errors[0]
        error_type = first_error.get('type', '')
        field = first_error.get('loc', [])[-1] if first_error.get('loc') else 'field'
        
        if error_type == 'string_too_long':
            max_length = first_error.get('ctx', {}).get('max_length', 'unknown')
            current_length = len(first_error.get('input', ''))
            message = f"Field '{field}' exceeds maximum length of {max_length} characters (current: {current_length})"
        else:
            message = first_error.get('msg', 'Validation error')
            message = f"{field}: {message}"
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": [{
                    "msg": message,
                    "type": error_type,
                    "loc": first_error.get('loc', [])
                }]
            },
        )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


app.add_middleware(APIKeyMiddleware)

#root endpoint
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {"message": "Ollama Bridge Server is running"}

app.include_router(models_route.router, tags=["models"])
app.include_router(chat.router, tags=["chat"])
app.include_router(chat_stream.router, tags=["chat"])

app.include_router(chat_debug.router, tags=["debug"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
