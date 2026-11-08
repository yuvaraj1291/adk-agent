"""RAG retrieval against a Vertex AI Vector Search index endpoint.

Each datapoint in the index stores the chunk text and its source in
``embedding_metadata`` (written by ``scripts/ingest.py``), so a single
``find_neighbors`` call returns everything the agent needs.

Indexes built elsewhere may store only vectors and IDs. For those, set
``CHUNKS_URI`` to a JSONL file with one ``{"id": ..., "text": ..., ...}`` row
per datapoint, and results are filled in from it by ID.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any
from urllib.parse import urlparse

from google import genai
from google.cloud import aiplatform
from google.genai import types


def _env(name: str, default: str | None = None) -> str:
  value = os.getenv(name, default)
  if not value:
    raise RuntimeError(
        f"Environment variable {name} is not set. See rag_agent/.env.example."
    )
  return value


def _vs_location() -> str:
  """Region of the Vector Search index (and embedding model).

  Kept separate from GOOGLE_CLOUD_LOCATION, which ADK uses for the Gemini
  model and may be "global".
  """
  return os.getenv("VECTOR_SEARCH_LOCATION") or "us-central1"


@functools.lru_cache(maxsize=1)
def _genai_client() -> genai.Client:
  return genai.Client(
      vertexai=True,
      project=_env("GOOGLE_CLOUD_PROJECT"),
      location=_vs_location(),
  )


@functools.lru_cache(maxsize=1)
def _index_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
  aiplatform.init(project=_env("GOOGLE_CLOUD_PROJECT"), location=_vs_location())
  return aiplatform.MatchingEngineIndexEndpoint(
      index_endpoint_name=_env("VECTOR_SEARCH_INDEX_ENDPOINT")
  )


def _parse_gcs_uri(uri: str) -> tuple[str, str] | None:
  """Returns (bucket, blob) for gs:// or storage.googleapis.com URLs."""
  parsed = urlparse(uri)
  if parsed.scheme == "gs":
    return parsed.netloc, parsed.path.lstrip("/")
  if parsed.scheme in ("http", "https") and parsed.netloc == "storage.googleapis.com":
    bucket, _, blob = parsed.path.lstrip("/").partition("/")
    return bucket, blob
  return None


def _read_chunks_file(uri: str) -> str:
  gcs = _parse_gcs_uri(uri)
  if gcs is None:  # Local file.
    with open(uri, encoding="utf-8") as f:
      return f.read()
  from google.cloud import storage  # Installed with google-cloud-aiplatform.
  bucket, blob = gcs
  client = storage.Client(project=_env("GOOGLE_CLOUD_PROJECT"))
  return client.bucket(bucket).blob(blob).download_as_text()


@functools.lru_cache(maxsize=1)
def _chunk_store() -> dict[str, dict[str, Any]]:
  """Loads CHUNKS_URI once and maps datapoint ID -> chunk row ({} if unset)."""
  uri = os.getenv("CHUNKS_URI")
  if not uri:
    return {}
  store = {}
  for line in _read_chunks_file(uri).splitlines():
    if line.strip():
      row = json.loads(line)
      store[str(row["id"])] = row
  return store


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
  """Embeds texts with the configured Vertex AI embedding model."""
  response = _genai_client().models.embed_content(
      model=_env("EMBEDDING_MODEL", "text-embedding-005"),
      contents=texts,
      config=types.EmbedContentConfig(
          task_type=task_type,
          output_dimensionality=int(_env("EMBEDDING_DIMENSIONS", "768")),
      ),
  )
  return [e.values for e in response.embeddings]


def search_knowledge_base(query: str) -> dict[str, Any]:
  """Searches the private knowledge base (Vertex AI Vector Search) for passages relevant to the query.

  Use this for questions about internal / proprietary documents that were
  ingested into the knowledge base.

  Args:
    query: A natural-language search query describing the information needed.

  Returns:
    A dict with ``status`` and, on success, a list of ``results`` each holding
    ``text``, ``source`` and ``distance`` (higher means more similar).
  """
  try:
    [query_vector] = embed_texts([query], task_type="RETRIEVAL_QUERY")
    neighbors = _index_endpoint().find_neighbors(
        deployed_index_id=_env("VECTOR_SEARCH_DEPLOYED_INDEX_ID"),
        queries=[query_vector],
        num_neighbors=int(_env("VECTOR_SEARCH_TOP_K", "5")),
        return_full_datapoint=True,
    )
    chunks = _chunk_store()
  except Exception as e:  # Surface errors to the model instead of crashing.
    return {"status": "error", "error_message": f"{type(e).__name__}: {e}"}

  results = []
  for n in neighbors[0] if neighbors else []:
    metadata = dict(n.embedding_metadata or {})
    chunk = chunks.get(str(n.id), {})
    result = {
        "id": n.id,
        "text": metadata.pop("text", "") or chunk.get("text", ""),
        "source": metadata.pop("source", "") or chunk.get("source", ""),
        "distance": n.distance,
    }
    if chunk.get("page") is not None:
      result["page"] = chunk["page"]
    if metadata:  # Any other metadata stored on the datapoint.
      result["metadata"] = metadata
    if n.restricts:
      result["restricts"] = {
          r.name: list(r.allow_tokens or []) for r in n.restricts
      }
    results.append(result)
  if not results:
    return {"status": "no_results", "results": []}
  return {"status": "success", "results": results}
