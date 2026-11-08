"""RAG retrieval against a Vertex AI Vector Search index endpoint.

Each datapoint in the index stores the chunk text and its source in
``embedding_metadata`` (written by ``scripts/ingest.py``), so a single
``find_neighbors`` call returns everything the agent needs.
"""

from __future__ import annotations

import functools
import os
from typing import Any

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
  except Exception as e:  # Surface errors to the model instead of crashing.
    return {"status": "error", "error_message": f"{type(e).__name__}: {e}"}

  results = []
  for n in neighbors[0] if neighbors else []:
    metadata = dict(n.embedding_metadata or {})
    result = {
        "id": n.id,
        "text": metadata.pop("text", ""),
        "source": metadata.pop("source", ""),
        "distance": n.distance,
    }
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
