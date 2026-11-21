# LangGraph Chat Bot

A small learning project: a chatbot built with **LangGraph** + **LangChain** + **Groq**, exposed through a **FastAPI** backend and a **Streamlit** frontend. Conversations are persisted per thread in SQLite, the bot can call tools (calculator, web search, RAG over a PDF), summarize long conversations, and supports human-in-the-loop approval for tool use.

## Features

- **LangGraph state machine** — `chat_bot` node + `tools` node, routed with `tools_condition`.
- **Persistent threads** — `SqliteSaver` checkpointer stores state in `chatbot.db`, so conversations can be resumed by `thread_id`.
- **Tool calling** — calculator, DuckDuckGo search, and a RAG tool over `Resources/MCP_Server.pdf`.
- **Human-in-the-loop** — the calculator tool uses `interrupt` to ask for approval before running.
- **Streaming responses** — FastAPI streams tokens via `StreamingResponse` (`stream_mode="messages"`).
- **Conversation summarization** — when a thread grows past 10 messages, an LLM summary is generated and prepended as context.
- **Message trimming** — `trim_messages` keeps the prompt within a token budget.
- **RAG** — PDF → chunks → HuggingFace embeddings → Qdrant (local, `qdrant_db/`).
- **Streamlit UI** — chat interface, thread list, backend URL setting, tool-approval prompt.

## Tech stack

- Python 3.11
- LangGraph 1.2, LangChain 1.x, LangChain-Groq
- Groq model: `openai/gpt-oss-20b`
- FastAPI + Uvicorn
- Streamlit
- SQLite (checkpointing), Qdrant (vector store)
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` embeddings
- DuckDuckGo search

## Project structure

```
chat_bot_langgraph/
├── main.py                  # FastAPI app + LangGraph graph + checkpointing + endpoints
├── streamlit_frontend.py    # Streamlit chat UI
├── tools_file.py            # Tool definitions (calculator, ducksearch, rag) + bound LLM
├── summarise.py             # Conversation summarization logic
├── rag_tool.py              # RAG tool: PDF loading, chunking, Qdrant retrieval
├── Longterm.py              # Standalone exploration of LangGraph InMemoryStore (long-term memory)
├── requirements.txt         # Pinned dependencies
├── .env                     # API keys (not committed)
├── .gitignore
├── Resources/               # Source PDF for RAG (ignored)
└── qdrant_db/               # Local Qdrant storage (generated)
```

## Prerequisites

- Python 3.11
- A Groq API key
- (Optional) LangSmith API key for tracing
- (Optional) HuggingFace token for embeddings download

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Create a .env file with:
# GROQ_API_KEY=...
# LANGCHAIN_TRACING_V2=true            (optional)
# LANGCHAIN_ENDPOINT=...               (optional)
# LANGCHAIN_API_KEY=...                (optional)
# LANGCHAIN_PROJECT=chatbot_langgraph   (optional)
# HUGGINGFACE_API_KEY=...              (optional)
```

## Running

Start the backend first, then the frontend.

```bash
# Terminal 1 - backend (port 8080)
uvicorn main:app --reload --port 8080

# Terminal 2 - frontend
streamlit run streamlit_frontend.py
```

Open the Streamlit URL (usually `http://localhost:8501`). Set the backend URL in the sidebar if needed (default `http://localhost:8080`).

## API

### `POST /chat`

Request body:

```json
{
  "human_message": "What is MCP?",
  "thread_id": "optional-uuid"
}
```

- If `thread_id` is omitted, a new UUID thread is created.
- Response is a streamed plain-text reply.

### `GET /history/{thread_id}`

Returns the stored messages for a thread:

```json
{
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

## How it works (concepts)

- **State**: `WokflowState` is a `TypedDict` with `messages` (reduced by `add_messages`) and `summary`.
- **Graph**: `START -> chat_bot -> (tools_condition) -> tools -> chat_bot -> END`.
- **Checkpointing**: `SqliteSaver` persists state per `thread_id`, enabling resume and thread listing via `retrive_all_threads()`.
- **Tools**: defined with `@tool`, collected into `toollist`, bound to the LLM with `bind_tools`, executed by `ToolNode`.
- **HITL**: the calculator calls `interrupt(...)`; the graph pauses until resumed with a `Command`.
- **Streaming**: `workflow.stream(..., stream_mode="messages")` yields `(chunk, metadata)`; only chunks from the `chat_bot` node with content are yielded.
- **Summarization**: `summarise_stlm` runs inside `chatbot_func`; if no summary exists and messages exceed 10, it generates one and appends a summary message.
- **Trimming**: `trim_messages(strategy="last", max_tokens=200)` keeps recent context within budget.
- **RAG**: `PyPDFLoader` → `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` → `QdrantVectorStore` (local) → similarity retriever (k=3).

## Notes

- `Longterm.py` is a standalone script exploring LangGraph's `InMemoryStore` (long-term memory). It is not wired into the main graph.
- `Resources/`, `venv/`, `.env`, and `.claude/` are git-ignored.
- `improvements.md` (personal learning notes) is also git-ignored.

## Learning improvements

See `improvements.md` for a list of small, concept-deepening tweaks you can try. It is intentionally not committed.
