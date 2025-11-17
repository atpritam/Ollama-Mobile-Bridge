# Ollama Mobile Bridge: Intelligent Agentic LLM Orchestration

FastAPI backend that transforms local LLMs into intelligent agents with real-time web access, autonomous search routing, and sophisticated context management. Built to maximize the capabilities of models of any size through dynamic resource allocation and multi-step reasoning.

Model-agnostic architecture, supports all models available through Ollama ([See Models](https://ollama.ai/models)). System automatically adjusts based on model capability (3B vs 70B).

## Core Features

- **RAG Pipeline**: Multi-step reasoning process, detects if query needs fresh data, generates its own search queries, and synthesizes information for a final answer.
- **Tool Calling**: Detects user intent and routes to specialized tools (Google, Reddit, Wikipedia, Weather).
- **Streaming API**: Delivers responses using Server-Sent Events (SSE), providing real-time status updates and token-by-token streaming.
- **Context Management**: Automatically manages the LLM's context window, truncating conversation history to prevent overflow while dynamically making space for incoming search results.
- **User Memory & Personalization**: Allows for personalized interactions by providing the LLM with persistent user context.
- **Secure & Asynchronous**: Built on FastAPI for high-performance async operations and secured with an effective `API key middleware`.
- **Concurrent web scraping**: Parallel async fetching with type-specific strategies and sequential fallback when fetch returns empty content.
- **Model-Adaptive**: Automatically detects model size and adjusts search depth, content limits, and reasoning strategies accordingly.

## Tech Stack

  **LLM & AI:**
  - LLM Orchestration (Ollama)
  - RAG (Retrieval-Augmented Generation)
  - Agentic AI (autonomous tool selection & self-healing)

  **APIs & Data:**
  - Brave Search API, Wikipedia API, OpenWeatherMap API
  - Web scraping (httpx async, Beautiful Soup via custom parser)
  - Jina Reader API (JavaScript-rendered content fallback)

  **Backend:**
  - FastAPI (async Python web framework)
  - Server-Sent Events (SSE streaming)
  - Pydantic (data validation)
  - Asyncio (concurrent operations)

  **Production:**
  - Token management & context window optimization
  - Security (API auth, content validation, rate limiting)
  - Error handling & fallback mechanisms

## Architecture

```
Ollama-Mobile-Bridge/
├── main.py                      # Application entry point
├── auth.py                      # Authentication middleware
├── config.py                    # Dynamic configuration & model detection
├── models/
│   ├── api_models.py            # Pydantic API request/response models
│   └── chat_models.py           # Internal data structures
├── routes/
│   ├── chat.py                  # Chat endpoints (standard & streaming)
│   └── models_route.py          # Model listing endpoints
├── services/
│   ├── chat_service.py          # Core orchestration logic & decision flow
│   ├── search.py                # Brave Search API integration & web scraping
│   └── weather.py               # OpenWeatherMap integration
└── utils/
    ├── constants.py             # System prompts and constants
    ├── html_parser.py           # HTML parsing & content extraction
    ├── logger.py                # Logging configuration
    └── token_manager.py         # Context window and token management
```

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running locally
- Brave Search API key (optional, for web search)
- OpenWeatherMap API key (optional, for weather data)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/atpritam/Ollama-Mobile-Bridge.git
    cd Ollama-Mobile-Bridge
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables**:
    ```bash
    cp .env.example .env
    ```

4.  **Configure API keys** in `.env`:
    ```env
    BRAVE_SEARCH_API_KEY=your_brave_search_key
    OPENWEATHER_API_KEY=your_openweather_api_key
    API_KEY=your_app_api_key
    ```

   - **Brave Search API**: Get a free API key at [Brave Search API](https://brave.com/search/api/)
   - **OpenWeatherMap API**: Get a free API key at [OpenWeatherMap API](https://openweathermap.org/api)

## Quick Start

1.  **Ensure Ollama is running**:
    ```bash
    ollama serve
    ```

2.  **Start the server**:
    ```bash
    python main.py
    ```

3.  The server will start on `http://0.0.0.0:8000`.

## API Endpoints

### List Available Models
```http
GET /list
X-API-Key: app_api_key
```
Returns all Ollama models available on your system.

### Standard Chat
```http
POST /chat
Content-Type: application/json
X-API-Key: app_api_key

{
  "model": "llama3.2:3b",
  "prompt": "What is the latest news on it?",
  "history": [
    {
      "role": "user",
      "content": "Whatis the Artemis program?"
    },
    {
      "role": "assistant",
      "content": "The Artemis program is a NASA mission aimed at returning humans to the lunar surface."
    }
  ]
}
```

### Streaming Chat
```http
POST /chat/stream
Content-Type: application/json
X-API-Key: app_api_key

{
  "model": "llama3.2:3b",
  "prompt": "Recommend a good sci-fi book to read.",
  "user_memory": "I live in Boston. I have already read 'Dune' and 'The Expanse' series."
}
```

The streaming endpoint returns Server-Sent Events:
- `event: status`: The current stage (`thinking`, `searching_google`, `generating`).
- `event: token`: A piece of the response.
- `event: done`: The final metadata object.

### Example API Response
```json
{
  "model": "llama3.2:3b",
  "context_messages_count": 3,
  "search_performed": true,
  "tokens": {
    "used": 130,
    "limit": 98304,
    "model_max": 131072,
    "usage_percent": 0.1
  },
  "search_type": "google",
  "search_query": "latest news Artemis program",
  "source": "https://www.nasa.gov/artemis-i/",
  "response": "The Artemis II mission is scheduled for April 2026..."
}
```

## Contributing

Contributions are welcome!

## Support

For issues, questions, or contributions, please [open an issue](https://github.com/atpritam/Ollama-Mobile-Bridge/issues).