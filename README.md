# ChatLocalLLM: An Agentic Backend for Local LLMs

FastAPI-based application that serves as an intelligent backend, augmenting local models with real-time web search, weather information, and a persistent memory, all orchestrated through a sophisticated, multi-step Retrieval-Augmented Generation pipeline.

It's designed to be a robust, private, and powerful engine for building applications on top of local LLMs.

## Core Features

- **RAG Pipeline**: The system uses a multi-step reasoning process to decide if a query needs fresh data, generates its own search queries, and synthesizes information for a final answer.
- **Tool Use**: Automatically detects the user's intent and routes queries to the appropriate tool, including Google Search, Reddit, Wikipedia, and a live Weather API.
- **Streaming API**: Delivers responses using Server-Sent Events (SSE), providing real-time status updates and token-by-token streaming.
- **Context Management**: Automatically manages the LLM's context window, truncating conversation history to prevent overflow while dynamically making space for incoming search results.
- **User Memory & Personalization**: A `user_memory` field allows for personalized interactions by providing the LLM with persistent user context (e.g., location, preferences).
- **Secure & Asynchronous**: Built on FastAPI for high-performance async operations and secured with a simple and effective API key middleware.

## Agentic RAG Pipeline

The heart of this project is its decision-making engine. The service uses a series of steps to reason about the user's request.

**Orchestration Flow:**
1. **Pre-flight Check**: First, the system analyzes the query for a keyword combination pattern indicating a need for recent information.
2. **LLM Call #1 (Reasoning Step)**:
    - If recency is suspected, the LLM is prompted to generate a precise search query.
    - Otherwise, it attempts to answer the query directly from its own knowledge.
3. **Response Analysis**: The initial response is analyzed for "knowledge cutoff" phrases and if detected, the flow is re-routed to generate a search query.
4. **Dynamic Search Execution**: The generated query is run against the appropriate tool (Google, Weather, etc.).
5. **LLM Call #2 (Synthesis Step)**: The search results are injected into a new, specialized prompt, and the LLM is called a final time to generate a conversational answer based on the retrieved information.

---

## Prerequisites

- Python 3.11+
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
X-API-Key: your_app_api_key
```
Returns all Ollama models available on your system.

### Standard Chat
```http
POST /chat
Content-Type: application/json
X-API-Key: your_app_api_key

{
  "model": "llama3:8b",
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
X-API-Key: your_app_api_key

{
  "model": "llama3:8b",
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
  "model": "llama3:8b",
  "context_messages_count": 3,
  "search_performed": true,
  "tokens": {
    "used": 137,
    "limit": 7900,
    "model_max": 8192,
    "usage_percent": 1.7
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