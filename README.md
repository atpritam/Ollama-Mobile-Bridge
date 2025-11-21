# Ollama Mobile Bridge: Intelligent Agentic LLM Orchestration

FastAPI backend that transforms local LLMs into fully capable agentic systems with real-time web access, autonomous search routing, and advanced context management. Designed to maximize the capabilities of models of any size through dynamic strategy selection and multi-step reasoning.

The architecture is fully model-agnostic, supporting all Ollama-compatible models ([See List](https://ollama.ai/models)).
The system automatically adjusts reasoning depth and tool usage based on the model’s capability (e.g., 3B vs 70B).

## Core Features

- **RAG Pipeline**: Performs multi-step reasoning, determines if fresh data is needed, generates search queries, performs external search or internal recall, and synthesizes results into a final answer.
- **Tool Calling**: Detects user intent and routes to specialized tools (Google, Reddit, Wikipedia, Weather, Recall).
- **Concurrent web scraping**: High-performance parallel fetching with type-specific extraction and fallback URLs when content is empty.
- **Smart Caching**: Multi-level caching for queries and URLs using Jaccard/Cosine similarity, SimHash, and WordNet synonyms to avoid redundant searches and scraping.
- **Memory & Recall**: The agent can reference its own past search results within a conversation, enabling more coherent and context-aware follow-up responses.
- **Context Management**: Automatically manages the model’s context window, truncating history while dynamically making space for incoming search results.
- **User Memory**: Allows for personalized interactions by providing the LLM with persistent user context.
- **Streaming API**: Delivers responses using Server-Sent Events (SSE), providing real-time status updates and token-by-token streaming.

## Tech Stack

  **LLM & AI:**
  - LLM (Ollama Local & Cloud models)
  - RAG (Retrieval-Augmented Generation)
  - Agentic AI (autonomous tool selection & self-healing)

  **APIs & Data:**
  - Brave Search API, Wikipedia API, OpenWeatherMap API
  - Web scraping (httpx async, Beautiful Soup via custom parser)
  - Jina Reader API (JavaScript-rendered content fallback)

  **Backend:**
  - Server-Sent Events (SSE streaming)
  - Httpx connection pooling
  - Pydantic (data validation)
  - Asyncio (concurrent operations)
  - SQLite (persistent cache)

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
├── tests/                       # Unit and integration tests
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
    ├── http_client.py           # httpx connection pooling
    ├── token_manager.py         # Context window and token management
    ├── cache.py                 # Search result caching & similarity detection
    └── text_similarity.py       # Query similarity algorithms
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

2.  **Activate virtual environment**:
    ```bash
    python3.14 -m venv .venv    # use appropriate Python version
    source .venv/bin/activate   # Activate virtual environment
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    ```bash
    cp .env.example .env
    ```

5.  **Configure API keys** in `.env`:
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
    # Install Ollama: https://ollama.com/download
    ollama serve 
    ```

2.  **Start the server**:
    ```bash
    python main.py
    ```

3.  The server will start on `http://0.0.0.0:8000`.

## Run Tests
```bash
 python -m pytest # 85 tests
```

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
  "search_id": 2,
  "response": "The Artemis II mission is scheduled for April 2026..."
}
```

## Contributing

Contributions are welcome!

## Support

For issues, questions, or contributions, please [open an issue](https://github.com/atpritam/Ollama-Mobile-Bridge/issues).