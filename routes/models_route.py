"""
Route handlers for model listing operations.
"""
from fastapi import APIRouter
import ollama

router = APIRouter()

@router.get("/list")
async def list_models():
    """List all locally available Ollama models."""
    try:
        client = ollama.AsyncClient()
        models_response = await client.list()
        models = [model['model'] for model in models_response['models']]
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}