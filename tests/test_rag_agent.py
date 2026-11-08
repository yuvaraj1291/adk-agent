"""Offline tests (no GCP calls). Run with: pytest -q"""

from types import SimpleNamespace
from unittest import mock

import pytest

from rag_agent import agent, vector_search
from scripts import ingest


@pytest.fixture(autouse=True)
def env(monkeypatch):
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  monkeypatch.setenv("VECTOR_SEARCH_INDEX_ENDPOINT", "123")
  monkeypatch.setenv("VECTOR_SEARCH_DEPLOYED_INDEX_ID", "deployed")
  vector_search._genai_client.cache_clear()
  vector_search._index_endpoint.cache_clear()


def test_agent_has_both_tools():
  tool_names = [getattr(t, "name", getattr(t, "__name__", None))
                for t in agent.root_agent.tools]
  assert tool_names == ["google_search", "search_knowledge_base"]
  assert agent.root_agent.tools[0].bypass_multi_tools_limit


@pytest.mark.asyncio
async def test_google_search_is_wrapped_as_agent_tool():
  from google.adk.agents.readonly_context import ReadonlyContext
  ctx = mock.MagicMock(spec=ReadonlyContext)
  tools = await agent.root_agent.canonical_tools(ctx)
  types_ = {type(t).__name__ for t in tools}
  assert "GoogleSearchAgentTool" in types_
  assert "FunctionTool" in types_


def test_search_knowledge_base_success():
  neighbor = SimpleNamespace(
      id="abc", distance=0.87, restricts=None,
      embedding_metadata={"text": "31 days PTO", "source": "handbook.md"})
  endpoint = mock.MagicMock()
  endpoint.find_neighbors.return_value = [[neighbor]]
  with mock.patch.object(vector_search, "embed_texts", return_value=[[0.1] * 768]) as emb, \
       mock.patch.object(vector_search, "_index_endpoint", return_value=endpoint):
    out = vector_search.search_knowledge_base("vacation days?")
  emb.assert_called_once_with(["vacation days?"], task_type="RETRIEVAL_QUERY")
  kwargs = endpoint.find_neighbors.call_args.kwargs
  assert kwargs["deployed_index_id"] == "deployed"
  assert kwargs["return_full_datapoint"] is True
  assert out == {"status": "success", "results": [
      {"id": "abc", "text": "31 days PTO", "source": "handbook.md", "distance": 0.87}]}


def test_search_knowledge_base_no_results_and_errors():
  endpoint = mock.MagicMock()
  endpoint.find_neighbors.return_value = [[]]
  with mock.patch.object(vector_search, "embed_texts", return_value=[[0.0]]), \
       mock.patch.object(vector_search, "_index_endpoint", return_value=endpoint):
    assert vector_search.search_knowledge_base("x")["status"] == "no_results"
  with mock.patch.object(vector_search, "embed_texts", side_effect=RuntimeError("boom")):
    out = vector_search.search_knowledge_base("x")
  assert out["status"] == "error" and "boom" in out["error_message"]


def test_chunking_respects_size():
  text = "\n\n".join(["word " * 50] * 20) + "\n\n" + "x" * 2500
  chunks = ingest.chunk_text(text)
  assert chunks and all(len(c) <= ingest.CHUNK_SIZE for c in chunks)


def test_build_datapoints_stores_metadata(tmp_path):
  f = tmp_path / "doc.md"
  f.write_text("Hello world.\n\nSecond paragraph.")
  with mock.patch.object(ingest, "embed_texts", side_effect=lambda t, task_type: [[0.5] * 3] * len(t)):
    dps = ingest.build_datapoints([f])
  assert len(dps) == 1
  assert dps[0].embedding_metadata["text"].startswith("Hello world.")
  assert dps[0].embedding_metadata["source"].endswith("doc.md")
  assert list(dps[0].feature_vector) == [0.5] * 3
