"""
Ollama Mobile Bridge - FastAPI application for chatting with local LLMs with web search integration.
"""
from fastapi import FastAPI
from config import Config
from routes import chat, models_route

app = FastAPI(title=Config.APP_TITLE)

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