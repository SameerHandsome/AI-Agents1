# AI-Agents1

This repository contains two example AI agent FastAPI applications and supporting tooling for evaluation and dataset generation. Everything in this README is derived directly from the repository code (see `basic_agent/`, `advanced_agent/`, `evaluator.py`, and `generate_golden_dataset.py`).

## Overview — What this project does

- Provides two agents implemented as FastAPI apps:
  - `basic_agent` — a minimal ReAct-style agent with conversation memory and tool calling.
    - Entry: `basic_agent/main.py`
    - Version: 1.0.0 (server prints indicate it runs on port `8001` by default)
  - `advanced_agent` — a LangGraph-based agent with many added capabilities (retry, budget limits, guardrails, parallel tools, reflection, output validation, planning, per-tool rate limiting).
    - Entry: `advanced_agent/main.py`
    - Version: 2.0.0 (server prints indicate it runs on port `8000` by default)

- Implements a common set of tools used by both agents:
  - `tavily_search` (web search via Tavily API)
  - `get_weather`
  - `convert_currency`
  - `get_wikipedia_summary`
  - `get_world_time`
  - Tool implementations:
    - Basic variants: `basic_agent/tools.py`
    - Advanced variants (with decorators for retry + validation): `advanced_agent/tools.py`

- Persists chat history and context in SQLite (`chat_history.db`) through MemoryManager classes:
  - Basic memory: `basic_agent/graph.py` (stores and returns recent messages)
  - Advanced memory: `advanced_agent/graph.py` (stores messages, generates and stores session summaries in `session_summaries` table)

- Evaluation and dataset utilities:
  - Golden dataset generator: `generate_golden_dataset.py` → outputs `golden_dataset.json`
  - Agent evaluator that compares basic and advanced agent outputs against the golden dataset: `evaluator.py`

## Architecture and logic flow

High-level components:
- FastAPI server(s) exposing endpoints:
  - `/` — root info
  - `/health` — health + feature list
  - `POST /chat` — main chat endpoint
  - `GET /history/{session_id}` — retrieve recent messages
  - `DELETE /history/{session_id}` — clear session history
  - Advanced-only: `GET /stats` — simple usage statistics
  - Files: `basic_agent/main.py`, `advanced_agent/main.py`

Agent internals:
- StateGraph (LangGraph) is used to define the agent's processing graph:
  - Basic graph (simple ReAct loop):
    - Node "chat" → may route to tools via `ToolNode(ALL_TOOLS)` when `tools_condition` triggers
    - Implemented in `basic_agent/graph.py` (function `create_agent`)
    - LLM is `ChatOpenAI` (from `langchain_openai`) bound to tools via `llm.bind_tools(ALL_TOOLS)`
    - MemoryManager provides recent context (last ~5 messages) from SQLite and is consulted before invoking the LLM.
  - Advanced graph (planner + chat + parallel tools + reflection):
    - Nodes: `planner`, `chat_node`, `tools` (parallel tool execution), `reflection`
    - Routing:
      - `route_complexity` decides if a complex query should go to `planner` or straight to `chat_node`
      - `planner` produces a step plan (uses `PLANNER_PROMPT` in `advanced_agent/config.py`)
      - `chat_node` runs the LLM, enforces budget limits (MAX_TOOL_CALLS, MAX_EXECUTION_TIME) and guardrails
      - `tools` executes multiple tool calls in parallel using a `ThreadPoolExecutor` and a `RateLimiter`
      - `reflection` performs self-critique and may trigger regeneration (limited by MAX_REFLECTIONS)
    - Implemented in `advanced_agent/graph.py` (function `create_agent`)
    - A `MemorySaver` checkpoint is used when compiling the graph; session summaries are stored in `session_summaries`.
- Tools:
  - Basic tools (no retry / no validation): `basic_agent/tools.py`
  - Advanced tools are decorated with:
    - `retry_on_failure(max_retries=3)` — exponential backoff
    - `validate_output(validator)` — returns a validation failure string if validation fails
    - Located in `advanced_agent/tools.py`
  - Both sets expose `ALL_TOOLS` list used by the graphs.

Memory and persistence:
- Both agents use an SQLite DB `chat_history.db`.
- Basic MemoryManager:
  - Creates `chat_history` table if missing
  - `add_message(session_id, role, content)`, `get_recent_messages(session_id, limit=5)`, `get_context(session_id)` (returns recent messages joined)
  - Files: `basic_agent/graph.py`
- Advanced MemoryManager:
  - Creates `chat_history` and `session_summaries`
  - `summarize_history(session_id, llm)` uses the LLM to produce a 3-4 sentence summary and stores it in `session_summaries`
  - `get_context(session_id)` will include `session_summaries` and recent messages
  - File: `advanced_agent/graph.py`

Guardrails, rate-limiting and validation (advanced):
- Guardrails: `check_guardrails(message)` checks incoming messages against BLOCKED_PATTERNS (in `advanced_agent/config.py`) to prevent prompt-injection-like patterns.
- Rate limiter: In-memory RateLimiter used to enforce per-tool limits defined in `RATE_LIMITS` (in `advanced_agent/config.py`).
- Budgeting: `MAX_TOOL_CALLS` and `MAX_EXECUTION_TIME` defined in `advanced_agent/config.py`.
- Reflection: `REFLECTION_PROMPT` defined in `advanced_agent/config.py` is used by the reflection node.

Evaluation:
- `generate_golden_dataset.py` contains a curated task list and writes `golden_dataset.json`.
- `evaluator.py` creates both agents in-process (imports `basic_agent.graph.create_agent` and `advanced_agent.graph.create_agent`) and invokes them with queries. It computes metrics (tool recall, precision, task success, etc.) and writes results to a timestamped JSON file.

## Tech stack (derived from imports)

Primary libraries referenced in code:
- FastAPI — HTTP server and API definitions (`fastapi`)
- Pydantic — request/response models (`pydantic`)
- uvicorn — used in `if __name__ == "__main__"` blocks to run servers
- LangGraph (`langgraph`) — StateGraph and prebuilt ToolNode, graph routing
- LangChain/Core messages and tools (`langchain_core.messages`, `langchain_core.tools`) and `langchain_openai.ChatOpenAI` — LLM integration and tool definitions
- Requests — external HTTP calls from tools
- python-dotenv — `load_dotenv()` in configs
- pytz — timezone handling in `get_world_time`
- sqlite3 (stdlib) — local persistence (chat_history.db)
- concurrent.futures (stdlib) — parallel tool execution (advanced agent)

Note: The project references provider-specific endpoints/config:
- LLM is configured to call Groq compatibility endpoint with these settings:
  - `GROQ_API_KEY` (env `OPENAI_API_KEY`), `GROQ_MODEL` = "meta-llama/llama-4-maverick-17b-128e-instruct"
  - `TAVILY_API_KEY` for the Tavily search tool
- Files: `basic_agent/config.py`, `advanced_agent/config.py`

## Configuration (env vars and constants)

- Environment variables used:
  - OPENAI_API_KEY — set to Groq/OpenAI-compatible key (used as `GROQ_API_KEY` in code)
  - TAVILY_API_KEY — used by `tavily_search` tool
- Advanced config constants (in `advanced_agent/config.py`):
  - GROQ_MODEL — default model string used by ChatOpenAI
  - MAX_TOKENS, MAX_TOOL_CALLS, MAX_EXECUTION_TIME, MAX_REFLECTIONS
  - RATE_LIMITS — per-tool rate limits for advanced agent
  - BLOCKED_PATTERNS — simple regex list for guardrails
  - REFLECTION_PROMPT, PLANNER_PROMPT, COMPLEX_QUERY_WORDS — prompts and thresholds used by advanced agent

## API Reference (derived from code)

Basic agent (`basic_agent/main.py`)
- GET / — root metadata
- GET /health — returns:
  - status, timestamp, features (["conversation_memory","tool_calling"]), database_status
- POST /chat
  - Request model `ChatRequest`:
    - `message: str`
    - `session_id: str = "default"`
  - Response model `ChatResponse`:
    - `response: str`
    - `session_id: str`
    - `timestamp: str`
  - Behavior: retrieves context using MemoryManager, invokes graph, writes user + assistant messages to DB.
- GET /history/{session_id}
  - Returns recent messages and count
- DELETE /history/{session_id}
  - Clears session history from DB

Advanced agent (`advanced_agent/main.py`)
- GET / — root metadata
- GET /health — returns:
  - status, timestamp, features (full feature list), database_status, rate_limits
- POST /chat
  - Request model `ChatRequest`:
    - same fields as basic
  - Response model `ChatResponse`:
    - `response: str`
    - `session_id: str`
    - `timestamp: str`
    - `metadata: dict` (extra information such as tool call counts, reflection count, etc. is populated by graph)
- GET /history/{session_id}
  - Returns messages and count (uses advanced MemoryManager; includes summarization support)
- DELETE /history/{session_id}
- GET /stats
  - Returns total_messages, total_sessions, recent_messages_24h and rate_limits

## How to run (based strictly on code)

1. Prerequisites (install required packages referenced by imports)
   - The code imports the following external packages; install them in a virtualenv:
     - fastapi, uvicorn, pydantic, python-dotenv, requests, pytz
     - langgraph, langchain_core, langchain_openai (used in code — these are referenced and must be installable in your environment)
   - Example (may need to adapt based on available package names/versions):
     - pip install fastapi uvicorn python-dotenv requests pytz
     - pip install langgraph langchain_core langchain_openai
     - If packages have different PyPI names in your environment, install the appropriate packages.

2. Set environment variables
   - At minimum:
     - export OPENAI_API_KEY="your-groq-key"
     - export TAVILY_API_KEY="your-tavily-key"
   - The code reads these via dotenv (both `basic_agent/config.py` and `advanced_agent/config.py`).

3. Start Basic Agent (example shown in `basic_agent/main.py`)
   - Option A: run module directly (script uses uvicorn in main guard)
     - python basic_agent/main.py
     - The script indicates it starts on port 8001 by default.
   - Option B: use uvicorn CLI (explicit)
     - uvicorn basic_agent.main:app --reload --port 8001

4. Start Advanced Agent (example shown in `advanced_agent/main.py`)
   - Option A:
     - python advanced_agent/main.py
     - The script indicates it starts on port 8000 by default.
   - Option B:
     - uvicorn advanced_agent.main:app --reload --port 8000

5. Example chat request (based on request model)
   - Basic:
     - curl -X POST "http://localhost:8001/chat" -H "Content-Type: application/json" -d '{"message":"What is the weather in Tokyo?","session_id":"demo"}'
   - Advanced:
     - curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message":"Search recent AI breakthroughs and summarize","session_id":"demo"}'

6. Run evaluation / dataset generation
   - Generate golden dataset:
     - python generate_golden_dataset.py
     - This writes `golden_dataset.json` (generator contains multiple example tasks).
   - Run evaluator (compares both agents using the golden dataset):
     - python evaluator.py
     - `evaluator.py` imports `basic_agent` and `advanced_agent` modules in-process, runs their graphs on dataset queries, computes metrics, and writes an output JSON (timestamped).

## Notable implementation details (code-derived)

- Both agents use `ChatOpenAI` (from `langchain_openai`) with:
  - `api_key=GROQ_API_KEY`
  - `base_url="https://api.groq.com/openai/v1"`
  - `model=GROQ_MODEL`
  - temperature = 0
  - See `basic_agent/graph.py` and `advanced_agent/graph.py`
- Tools are registered and bound to the LLM via `llm.bind_tools(ALL_TOOLS)`
- Advanced agent uses `StateGraph` routing:
  - `route_complexity` → `planner` or `chat_node`
  - `planner` → `chat_node` → conditional routing to tools or reflection → possibly back into `chat_node`
  - Graph compiled with `MemorySaver` checkpoint in advanced agent
  - See `advanced_agent/graph.py` for routing functions: `route_complexity`, `after_chat_routing`, `after_reflection_routing`
- Memory schemas:
  - `chat_history` table (both agents)
  - `session_summaries` (advanced agent only)
  - Created in `MemoryManager._init_db()` in the respective `graph.py`s
- Tool execution in advanced agent:
  - `parallel_tool_node` uses `ThreadPoolExecutor` to execute multiple tool calls concurrently while consulting `RateLimiter`
  - Each tool call returns a `ToolMessage` that gets fed back into the graph

## Where to look in the code

- Basic agent:
  - main: `basic_agent/main.py`
  - graph + memory: `basic_agent/graph.py`
  - tools: `basic_agent/tools.py`
  - config: `basic_agent/config.py`
- Advanced agent:
  - main: `advanced_agent/main.py`
  - graph + memory + routing: `advanced_agent/graph.py`
  - tools (with retry/validation): `advanced_agent/tools.py`
  - config: `advanced_agent/config.py`
- Evaluation and dataset:
  - `generate_golden_dataset.py`
  - `evaluator.py`

## Limitations & assumptions (explicit, code-derived)

- The repository expects specific external packages (e.g., `langgraph`, `langchain_core`, `langchain_openai`). Ensure these libraries are available in your environment — the code imports them directly.
- External APIs require keys: `OPENAI_API_KEY` and `TAVILY_API_KEY`. Tools that call external services will return errors or messages when API keys are missing.
- SQLite DB path is hard-coded as `chat_history.db` by default in both MemoryManagers.
- The advanced agent performs LLM-based summarization and reflection — this may cause additional LLM calls and costs.
