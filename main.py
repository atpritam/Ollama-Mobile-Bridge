"""
Ollama Mobile Bridge - FastAPI application for chatting with local LLMs.
Featuring agentic System with real-time web access, autonomous search routing, cache retrieval and advanced context management.
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from config import Config
from routes import chat, models_route
from auth import APIKeyMiddleware
from utils.http_client import HTTPClientManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    yield
    await HTTPClientManager.close_all()

app = FastAPI(title=Config.APP_TITLE, lifespan=lifespan)
app.add_middleware(APIKeyMiddleware)

#root endpoint
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {"message": "Ollama Bridge Server is running"}

app.include_router(models_route.router, tags=["models"])
app.include_router(chat.router, tags=["chat"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)