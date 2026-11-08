# ADK RAG + Google Search Agent

A simple [Google ADK](https://google.github.io/adk-docs/) agent with two tools:

| Tool | What it does |
|---|---|
| `google_search` | Gemini's built-in Google Search grounding, for public / current information. |
| `search_knowledge_base` | RAG over your own documents: embeds the query with Vertex AI (`text-embedding-005`) and calls `find_neighbors` on a **Vertex AI Vector Search** index endpoint. The chunk text and source are stored in each datapoint's `embedding_metadata`, so no extra document store is needed. |

Gemini's built-in Google Search can't share a request with function tools, so the agent uses
`GoogleSearchTool(bypass_multi_tools_limit=True)`. ADK then runs Google Search in a small
sub-agent automatically.

```
rag_agent/
  agent.py              # root_agent definition
  vector_search.py      # search_knowledge_base tool + embedding helper
  memory_tools.py       # remember_facts: saves to memory after the user confirms
  search_budget.py      # per-question search limits
  .env.example          # configuration template
scripts/
  setup_vector_search.py  # one-time: create index + public endpoint + deploy
  ingest.py               # chunk, embed and upsert .md/.txt files
data/sample/            # fictional sample doc for testing RAG
tests/                  # offline unit tests (no GCP calls)
```

## Prerequisites

- Python 3.10+
- A GCP project with billing and the **Vertex AI API** enabled
- [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in with Application Default Credentials:

```bash
gcloud auth application-default login
```

```bash
gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
```

You don't need an API key. Everything authenticates through ADC.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp rag_agent/.env.example rag_agent/.env
```

Then set `GOOGLE_CLOUD_PROJECT` in `rag_agent/.env`. If your project can't reach `AGENT_MODEL`
(you'll get a 404 "Publisher model ... not found"), change it to a Gemini model your project can use.

## 2. Create the Vector Search index (one time)

```bash
python scripts/setup_vector_search.py
```

This creates a streaming index (768 dims, dot product, unit-normalized), a public endpoint, and
deploys the index on one `e2-standard-2` replica. Deployment usually takes **20–40 minutes**.
The script writes `VECTOR_SEARCH_INDEX`, `VECTOR_SEARCH_INDEX_ENDPOINT` and
`VECTOR_SEARCH_DEPLOYED_INDEX_ID` into `rag_agent/.env`.

> **Cost:** a deployed index bills per node-hour while it is deployed, even when idle.
> Undeploy it when you're finished (see [Cleanup](#cleanup)).

**Using an existing index instead?** Skip this step and fill in the three `VECTOR_SEARCH_*`
values yourself. The index must be a stream-update index with the same dimensions as
`EMBEDDING_DIMENSIONS`. Its datapoints need `text` and `source` keys in `embedding_metadata`,
which `ingest.py` adds for you.

## 3. Ingest documents

```bash
python scripts/ingest.py
```

```bash
python scripts/ingest.py path/to/your/docs
```

The first command ingests `data/sample/`. The second ingests your own `.md`/`.txt` files.
Streaming upserts become searchable within seconds.

## 4. Run the agent

```bash
adk web
```

Run it from the repo root, open http://localhost:8000 and pick **rag_agent**. For a terminal chat instead:

```bash
adk run rag_agent
```

### Things to try

- **RAG:** "How many vacation days do Zephyrine Robotics employees get?" (expect 31, from `acme_handbook.md`)
- **RAG:** "What is the depth rating of the Kelpie-7?" (expect 450 m)
- **Google Search:** "What's the weather in Tromsø today?"
- **Both:** "Where is Zephyrine Robotics based, and what's the population of that city?"

In `adk web`, the **Events** tab shows which tool was called and what it returned. To see the exact
request sent to Gemini, select a model event and open the **Request** icon (third in the left icon bar).

### Persistent sessions (optional)

By default `adk web` keeps sessions locally. To store them in
[Vertex AI Agent Engine Sessions](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview)
instead, so conversations survive restarts and can be shared with a deployed agent, pass the same
Agent Engine instance used for memory:

```bash
adk web \
  --session_service_uri=agentengine://projects/PROJECT_NUMBER/locations/us-central1/reasoningEngines/ENGINE_ID \
  --memory_service_uri=agentengine://projects/PROJECT_NUMBER/locations/us-central1/reasoningEngines/ENGINE_ID
```

Agent Engine assigns session IDs, and sessions are stored per user ID.

### Long-term memory (optional)

The agent can remember facts about a user across sessions with
[Vertex AI Memory Bank](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview).
Before each model call, `PreloadMemoryTool` searches Memory Bank with the user's message and adds any
matches to the prompt.

Nothing is saved without the user's approval. When the user shares a lasting fact or preference, or
asks the agent to remember something, the agent calls `remember_facts` with the facts it wants to
keep. The first call only asks for confirmation (ADK's `request_confirmation`). In `adk web` this shows
a card that lists the facts as editable JSON. The user can delete or reword facts, then tick
**Confirmed** and **Submit**; submitting without ticking it saves nothing. Only the approved list is
written, through `memories.generate` with consolidation, so it merges with existing memories.

Point `adk web` at your Agent Engine instance with its full resource name, so the memory bank's
region is used even when `GOOGLE_CLOUD_LOCATION=global`:

```bash
adk web --memory_service_uri=agentengine://projects/PROJECT_NUMBER/locations/us-central1/reasoningEngines/ENGINE_ID
```

Memory Bank consolidates memories with the model set in its `memoryBankConfig.generationConfig`. If a
confirmed fact doesn't appear, check that your project can call that model in that location. Without
the flag, ADK falls back to an in-memory service that can't store facts directly, so `remember_facts`
returns an error.

### Search limits

Each question gets a small search budget so the agent can't keep searching: by default 3
`search_knowledge_base` calls and 2 web searches. Calls over the limit return `budget_exhausted`, and
once a tool's budget is spent it is removed from the next model request, so Gemini has to answer from
what it found or say it doesn't know. Change the limits in `rag_agent/.env`:

```
MAX_KNOWLEDGE_BASE_SEARCHES=3
MAX_WEB_SEARCHES=2
```

## Tests

```bash
pytest -q
```

The tests mock all GCP calls. They check the tool wiring, the retrieval result shape, error
handling, chunking, and datapoint building.

## Cleanup

In the Cloud Console go to **Vertex AI → Vector Search**. Undeploy the index from the endpoint,
then delete the endpoint and the index. You can also do it from Python:

```python
from google.cloud import aiplatform
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")
ep = aiplatform.MatchingEngineIndexEndpoint("ENDPOINT_RESOURCE_NAME")
ep.undeploy_all(); ep.delete()
aiplatform.MatchingEngineIndex("INDEX_RESOURCE_NAME").delete()
```
