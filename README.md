# NL for NoSQL — Natural Language Movie Query Agent

## Overview

NL for NoSQL is an AI-powered chatbot that lets users query a MongoDB NoSQL database using plain English. Built on the MongoDB Atlas `sample_mflix` dataset, the agent accepts natural language questions about movies and autonomously decides which combination of vector search, full-text search, and detailed document retrieval to use in order to construct an accurate answer. Conversation history and user memories are persisted back into MongoDB, making the system fully stateful across sessions.

The project was developed as an Advanced Database Systems assignment demonstrating how large language models can serve as a natural language interface on top of a NoSQL document store, bridging the gap between human intent and database queries without requiring the user to know any query language.

---

## Key Features

- **Natural Language Querying** — ask questions in plain English; the agent translates them into the appropriate database operations automatically.
- **Semantic Plot Search** — vector similarity search over movie plot embeddings surfaces films by concept or theme, not just keywords.
- **Full-Text Title Search** — Atlas full-text search index enables fast, fuzzy matching on movie titles.
- **Detailed Document Retrieval** — a dedicated tool fetches the full MongoDB document (cast, directors, genres, ratings, runtime, etc.) once a movie ID has been identified.
- **Persistent Conversation Memory** — `MongoDBSaver` checkpoints the entire LangGraph message state to MongoDB, so conversation context survives restarts and is resumable by session ID.
- **Long-Term User Memory** — `save_memory` / `retrieve_memories` tools backed by `MongoDBStore` with vector indexing let the agent remember user preferences across conversations.
- **Agentic ReAct Loop** — a LangGraph `StateGraph` orchestrates the agent→tools→agent cycle; the LLM decides at every step whether to call a tool or return a final answer.
- **Dual Interface** — a Streamlit web UI (`app.py`) with live tool-call status indicators, and a terminal CLI (`main.py`) for headless use.

---

## Applications

- **Movie Recommendation Systems** — users can describe the kind of film they want and receive semantically relevant suggestions.
- **Natural Language Database Interfaces** — a template for exposing any MongoDB collection to non-technical users via conversational AI.
- **Conversational Search Engines** — context-aware, multi-turn querying where follow-up questions refine rather than restart the search.
- **Knowledge Base Assistants** — the same architecture applies to any domain with a rich document store (products, research papers, legal documents, etc.).
- **Educational Demos** — illustrates how vector search, full-text search, and LLM reasoning can be combined in a single agentic pipeline.

---

## Future Work

- **Structured Query Generation** — extend the toolset to let the agent construct MongoDB aggregation pipelines for analytical queries (e.g., "top 10 highest-rated action movies from the 1990s").
- **Multi-Collection Support** — expose additional MongoDB collections (users, comments, sessions) and teach the agent to join information across them.
- **Hybrid Search** — combine vector and full-text scores using MongoDB Atlas's `$rankFusion` / `$vectorSearch` hybrid operators for improved retrieval precision.
- **User Profiles** — build richer long-term user profiles from saved memories to power personalised recommendations.
- **Evaluation Pipeline** — add an automated LLM-as-judge evaluation suite to measure retrieval quality and answer correctness across a benchmark question set.
- **Streaming Responses** — wire Streamlit's `st.write_stream` fully through the LangGraph event stream for truly token-level streaming.
- **Authentication** — add session-level authentication so memories and conversation history are scoped to individual users in a multi-tenant deployment.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- A MongoDB Atlas cluster with the `sample_mflix` dataset loaded and Network Access open to your IP
- An OpenAI API key

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd "nl for nosql"
uv sync
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-...
```

### 3. Run the Streamlit web app

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter a Session ID in the sidebar to persist conversation history, then ask questions like:

- *"Find me a sci-fi movie about artificial intelligence becoming self-aware."*
- *"What is the plot of Inception?"*
- *"Who directed The Godfather and who starred in it?"*

### 4. Run the CLI interface

```bash
uv run python main.py
```

Enter a session ID when prompted, then type your questions. Type `quit` to exit.

---

## Hardware & Software Configuration

| Component | Details |
|---|---|
| **OS** | Windows 11 Home Single Language (build 26200) |
| **CPU / RAM** | Standard laptop hardware (no GPU required) |
| **Python** | 3.12 (managed via `uv` + `.python-version`) |
| **Package manager** | `uv` 0.x with `pyproject.toml` / `uv.lock` |
| **LLM** | OpenAI `gpt-5.1` via `langchain-openai` |
| **Embedding model** | OpenAI `text-embeddings-small` (2048 dimensions) |
| **Database** | MongoDB Atlas (cloud) — `sample_mflix.embedded_movies` collection |
| **Vector index** | MongoDB Atlas Vector Search (`dotProduct` similarity, `plot_embedding` field) |
| **Full-text index** | MongoDB Atlas Search (`search_index` on `title` field) |
| **Agent framework** | LangGraph `StateGraph` with `MongoDBSaver` checkpointing |
| **Memory store** | LangGraph `MongoDBStore` with vector index (`sample_mflix.memories`) |
| **Web UI** | Streamlit 1.56 |
| **Key libraries** | `langchain>=1.2`, `langchain-mongodb>=0.11`, `langgraph>=1.1`, `pymongo>=4.15` |
