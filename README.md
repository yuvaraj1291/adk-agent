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

In `adk web`, the **Events** tab shows which tool was called and what it returned.

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
