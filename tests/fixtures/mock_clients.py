from unittest.mock import AsyncMock

class OllamaClientBuilder:
    """Factory for creating configurable Ollama mocks."""
    
    def __init__(self):
        self.responses = {}
        self.call_count = 0
    
    def set_response(self, call_num, content, stream=False):
        """Configure response for specific call number."""
        self.responses[call_num] = (content, stream)
        return self
    
    def build(self):
        """Build the AsyncMock."""
        mock_chat_method = AsyncMock()

        async def _mock_chat_side_effect(model, messages, stream=False, **kwargs):
            self.call_count += 1
            content, should_stream = self.responses.get(
                self.call_count, 
                ("default response", False)
            )
            
            if should_stream or stream:
                async def token_stream():
                    for token in content.split():
                        yield {"message": {"content": token + " "}}
                return token_stream()
            
            return {"message": {"content": content}}
        
        mock_chat_method.side_effect = _mock_chat_side_effect
        
        client = AsyncMock()
        client.chat = mock_chat_method
        return client

class FlexibleLLMClient:
    """LLM mock that tracks calls and supports sequences."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_history = []
    
    async def chat(self, model, messages, stream=False, **kwargs):
        self.call_history.append({
            "model": model,
            "messages": messages,
            "stream": stream
        })
        
        if stream:
            async def streamer():
                content = self.responses.pop(0) if self.responses else ""
                for chunk in content.split():
                    yield {"message": {"content": chunk + " "}}
            return streamer()
        
        return {
            "message": {"content": self.responses.pop(0) if self.responses else ""}
        }

