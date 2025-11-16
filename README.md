# Ollama Mobile Bridge

A FastAPI-based intelligent bridge that enhances local Ollama LLM models with real-time web search, weather information, and context-aware chat capabilities. This application automatically detects when queries require up-to-date information and augments LLM responses with current data from the web.

## Features

- **Intelligent Search Detection**: Automatically identifies queries requiring up-to-date information
- **Multi-Source Search**: Integrates Google search, Wikipedia, Reddit, and weather data
- **Streaming Responses**: Server-Sent Events (SSE) for real-time token-by-token responses
- **Context-Aware**: Maintains conversation history for coherent multi-turn dialogues
- **User Memory**: Memory for personalized responses (preferences, location, etc.)
- **Async Architecture**: Fully asynchronous for optimal performance

## Architecture

```
ChatLocalLLM/
├── main.py                      # Application entry point
├── auth.py                      # Authentication middleware.
├── config.py                    # Configuration and environment variables
├── models/
│   ├── api_models.py            # API request/response models
│   └── chat_models.py           # Internal data structures
├── routes/
│   ├── chat.py                  # Chat endpoints (standard & streaming)
│   └── models_route.py          # Model listing endpoints
├── services/
│   ├── chat_service.py          # Core orchestration logic
│   ├── search.py                # Brave Search API integration
│   └── weather.py               # OpenWeatherMap integration
└── utils/
    ├── constants.py             # System prompts and constants
    ├── html_parser.py           # HTML text extraction utilities
    └── logger.py                # Logging configuration
```

## Prerequisites

- Python 3.14+
- [Ollama](https://ollama.ai/) installed and running locally
- Brave Search API key (optional, for web search)
- OpenWeatherMap API key (optional, for weather data)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/atpritam/Ollama-Mobile-Bridge.git
   cd Ollama-Mobile-Bridge
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```

4. **Configure API keys** in `.env`:
   ```env
   BRAVE_SEARCH_API_KEY=your_brave_search_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   API_KEY=your_app_api_key
   ```

   - **Brave Search API**: Get free API key (2,000 queries/month) at [https://brave.com/search/api/](https://brave.com/search/api/)
   - **OpenWeatherMap API**: Get free API key (1,000 calls/day) at [https://openweathermap.org/api](https://openweathermap.org/api)

## Quick Start

1. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

2. **Start the server**:
   ```bash
   python main.py
   ```

3. **The server will start on** `http://0.0.0.0:8000`

## API Endpoints

### Health Check
```http
GET /
```
Returns basic health status.

### List Available Models
```http
GET /list
```
Returns all Ollama models available on your system with their details.

### Standard Chat
```http
POST http://127.0.0.1:8000/chat
Content-Type: application/json
X-API-Key: API_KEY

{
  "model": "llama3.2:3b",
  "prompt": "What's the weather like in its capital city?",
  "history": [
    {
      "role": "user",
      "content": "Tell me about France"
    },
    {
      "role": "assistant",
      "content": "France is a country in Western Europe. Its capital is Paris."
    }
  ]
}
```

### Streaming Chat
```http
POST http://127.0.0.1:8000/chat/stream
Content-Type: application/json
X-API-Key: API_KEY

{
  "model": "llama3.2:3b-instruct-q4_K_M",
  "prompt": "Recommend me a good restaurant for tonight",
  "user_memory": "I live in Boston. I have a nut allergy. I don't like to go out on rain."
}
```

Returns Server-Sent Events with real-time updates:
- `status`: Current processing stage (initializing, thinking, searching, generating)
- `search`: Search results and metadata
- `token`: Individual response tokens
- `done`: Final response and metadata

### API Response
```json
{
  "model": "llama3.2:3b",
  "context_messages_count": 3,
  "search_performed": true,
  "search_type": "weather",
  "search_query": "Paris",
  "source": "https://openweathermap.org",
  "response": "llm response"
}
```

## Development

### Project Structure

- **models/**: Pydantic data models for type safety
- **routes/**: FastAPI route handlers
- **services/**: Core business logic (chat orchestration, search, weather)
- **utils/**: Helper utilities (HTML parsing, logging, constants)

## Contributing

Contributions are welcome!

## Support

For issues, questions, or contributions, please [open an issue](https://github.com/atpritam/Ollama-Mobile-Bridge/issues).