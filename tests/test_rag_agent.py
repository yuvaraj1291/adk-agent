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
  monkeypatch.delenv("CHUNKS_URI", raising=False)
  vector_search._genai_client.cache_clear()
  vector_search._index_endpoint.cache_clear()
  vector_search._chunk_store.cache_clear()
  yield
  vector_search._chunk_store.cache_clear()


def test_agent_has_both_tools():
  tool_names = [getattr(t, "name", getattr(t, "__name__", None))
                for t in agent.root_agent.tools]
  assert tool_names == ["google_search", "search_knowledge_base",
                        "remember_facts", "preload_memory"]
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


def test_search_fills_text_from_chunks_file(tmp_path, monkeypatch):
  chunks = tmp_path / "chunks.jsonl"
  chunks.write_text(
      '{"id": "abc", "text": "Offer: 31 days PTO", "source": "gs://b/offer.pdf", "page": 2}\n'
      '\n'
      '{"id": "xyz", "text": "other", "source": "gs://b/other.pdf", "page": 1}\n')
  monkeypatch.setenv("CHUNKS_URI", str(chunks))
  vector_only = SimpleNamespace(id="abc", distance=0.6, restricts=None,
                                embedding_metadata=None)
  unknown = SimpleNamespace(id="missing", distance=0.5, restricts=None,
                            embedding_metadata=None)
  endpoint = mock.MagicMock()
  endpoint.find_neighbors.return_value = [[vector_only, unknown]]
  with mock.patch.object(vector_search, "embed_texts", return_value=[[0.1]]), \
       mock.patch.object(vector_search, "_index_endpoint", return_value=endpoint):
    out = vector_search.search_knowledge_base("vacation days?")
  assert out["results"] == [
      {"id": "abc", "text": "Offer: 31 days PTO", "source": "gs://b/offer.pdf",
       "distance": 0.6, "page": 2},
      {"id": "missing", "text": "", "source": "", "distance": 0.5}]


def test_datapoint_metadata_wins_over_chunks_file(tmp_path, monkeypatch):
  chunks = tmp_path / "chunks.jsonl"
  chunks.write_text('{"id": "abc", "text": "stale", "source": "old.md"}\n')
  monkeypatch.setenv("CHUNKS_URI", str(chunks))
  neighbor = SimpleNamespace(id="abc", distance=0.9, restricts=None,
                             embedding_metadata={"text": "fresh", "source": "new.md"})
  endpoint = mock.MagicMock()
  endpoint.find_neighbors.return_value = [[neighbor]]
  with mock.patch.object(vector_search, "embed_texts", return_value=[[0.1]]), \
       mock.patch.object(vector_search, "_index_endpoint", return_value=endpoint):
    [result] = vector_search.search_knowledge_base("q")["results"]
  assert (result["text"], result["source"]) == ("fresh", "new.md")


@pytest.mark.parametrize("uri, expected", [
    ("gs://bucket/a/b.jsonl", ("bucket", "a/b.jsonl")),
    ("https://storage.googleapis.com/bucket/a/b.jsonl", ("bucket", "a/b.jsonl")),
    ("/local/chunks.jsonl", None),
])
def test_parse_gcs_uri(uri, expected):
  assert vector_search._parse_gcs_uri(uri) == expected


def test_chunks_file_error_is_reported(monkeypatch):
  monkeypatch.setenv("CHUNKS_URI", "/does/not/exist.jsonl")
  endpoint = mock.MagicMock()
  endpoint.find_neighbors.return_value = [[]]
  with mock.patch.object(vector_search, "embed_texts", return_value=[[0.1]]), \
       mock.patch.object(vector_search, "_index_endpoint", return_value=endpoint):
    out = vector_search.search_knowledge_base("q")
  assert out["status"] == "error" and "FileNotFoundError" in out["error_message"]


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


# --- Search budget -----------------------------------------------------------

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402

from rag_agent import search_budget  # noqa: E402


def _budget_ctx(state, invocation_id="inv-1"):
  return SimpleNamespace(state=state, invocation_id=invocation_id)


def _tool(name):
  return SimpleNamespace(name=name)


def test_agent_wires_search_budget_callbacks():
  assert agent.root_agent.before_tool_callback is search_budget.limit_searches
  assert agent.root_agent.before_model_callback is search_budget.drop_spent_tools


def test_web_search_tool_name_matches_budget():
  from google.adk.agents.readonly_context import ReadonlyContext
  import asyncio
  tools = asyncio.run(agent.root_agent.canonical_tools(
      mock.MagicMock(spec=ReadonlyContext)))
  names = {t.name for t in tools}
  assert set(search_budget.search_limits()) <= names


def test_limit_searches_allows_up_to_limit_then_refuses():
  state = {}
  ctx = _budget_ctx(state)
  kb = _tool(search_budget.KNOWLEDGE_BASE_TOOL)
  web = _tool(search_budget.WEB_SEARCH_TOOL)
  assert [search_budget.limit_searches(kb, {}, ctx) for _ in range(3)] == [None] * 3
  refused = search_budget.limit_searches(kb, {}, ctx)
  assert refused["status"] == "budget_exhausted"
  # The web budget is separate.
  assert search_budget.limit_searches(web, {}, ctx) is None
  assert search_budget.limit_searches(web, {}, ctx) is None
  assert search_budget.limit_searches(web, {}, ctx)["status"] == "budget_exhausted"
  # Other tools are never limited.
  assert search_budget.limit_searches(_tool("preload_memory"), {}, ctx) is None


def test_limit_searches_resets_each_turn():
  state = {}
  kb = _tool(search_budget.KNOWLEDGE_BASE_TOOL)
  for _ in range(3):
    search_budget.limit_searches(kb, {}, _budget_ctx(state, "inv-1"))
  assert search_budget.limit_searches(kb, {}, _budget_ctx(state, "inv-2")) is None


def test_limits_come_from_env(monkeypatch):
  monkeypatch.setenv("MAX_KNOWLEDGE_BASE_SEARCHES", "1")
  monkeypatch.setenv("MAX_WEB_SEARCHES", "0")
  ctx = _budget_ctx({})
  assert search_budget.limit_searches(
      _tool(search_budget.KNOWLEDGE_BASE_TOOL), {}, ctx) is None
  assert search_budget.limit_searches(
      _tool(search_budget.KNOWLEDGE_BASE_TOOL), {}, ctx)["status"] == "budget_exhausted"
  assert search_budget.limit_searches(
      _tool(search_budget.WEB_SEARCH_TOOL), {}, ctx)["status"] == "budget_exhausted"


def _request():
  req = LlmRequest()
  req.config.tools = [types.Tool(function_declarations=[
      types.FunctionDeclaration(name=search_budget.KNOWLEDGE_BASE_TOOL),
      types.FunctionDeclaration(name=search_budget.WEB_SEARCH_TOOL),
  ])]
  return req


def _declared(req):
  return [d.name for t in (req.config.tools or [])
          for d in (t.function_declarations or [])]


def test_drop_spent_tools_leaves_request_alone_within_budget():
  req = _request()
  search_budget.drop_spent_tools(_budget_ctx({}), req)
  assert _declared(req) == [search_budget.KNOWLEDGE_BASE_TOOL,
                            search_budget.WEB_SEARCH_TOOL]
  assert not req.config.system_instruction


def test_drop_spent_tools_removes_only_spent_tool():
  state = {}
  ctx = _budget_ctx(state)
  for _ in range(3):
    search_budget.limit_searches(_tool(search_budget.KNOWLEDGE_BASE_TOOL), {}, ctx)
  req = _request()
  search_budget.drop_spent_tools(ctx, req)
  assert _declared(req) == [search_budget.WEB_SEARCH_TOOL]
  assert "used all your searches" in req.config.system_instruction


def test_drop_spent_tools_removes_all_tools_when_both_spent():
  ctx = _budget_ctx({})
  for name, n in [(search_budget.KNOWLEDGE_BASE_TOOL, 3),
                  (search_budget.WEB_SEARCH_TOOL, 2)]:
    for _ in range(n):
      search_budget.limit_searches(_tool(name), {}, ctx)
  req = _request()
  search_budget.drop_spent_tools(ctx, req)
  assert req.config.tools is None


# --- Confirmed memory writes -------------------------------------------------

from google.adk.tools.tool_confirmation import ToolConfirmation  # noqa: E402

from rag_agent import memory_tools  # noqa: E402


def _remember_ctx(confirmation=None):
  ctx = mock.MagicMock()
  ctx.tool_confirmation = confirmation
  ctx.add_memory = mock.AsyncMock()
  return ctx


def _saved_texts(ctx):
  memories = ctx.add_memory.call_args.kwargs["memories"]
  return [m.content.parts[0].text for m in memories]


def test_agent_no_longer_saves_automatically():
  assert agent.root_agent.after_agent_callback is None


@pytest.mark.asyncio
async def test_remember_facts_asks_first_and_saves_nothing():
  ctx = _remember_ctx()
  out = await memory_tools.remember_facts(
      ["Prefers  bullet points", "Prefers bullet points", " ", "Lives in Lisbon"], ctx)
  assert out == {"status": "awaiting_confirmation",
                 "facts": ["Prefers bullet points", "Lives in Lisbon"]}
  kwargs = ctx.request_confirmation.call_args.kwargs
  assert kwargs["payload"] == {"facts": ["Prefers bullet points", "Lives in Lisbon"]}
  assert "- Prefers bullet points" in kwargs["hint"]
  ctx.add_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_remember_facts_saves_after_confirmation():
  ctx = _remember_ctx(ToolConfirmation(confirmed=True,
                                       payload={"facts": ["Lives in Lisbon"]}))
  out = await memory_tools.remember_facts(["Lives in Lisbon"], ctx)
  assert out == {"status": "saved", "facts": ["Lives in Lisbon"]}
  assert _saved_texts(ctx) == ["Lives in Lisbon"]
  assert ctx.add_memory.call_args.kwargs["custom_metadata"] == {
      "enable_consolidation": True}
  ctx.request_confirmation.assert_not_called()


@pytest.mark.asyncio
async def test_remember_facts_saves_the_users_edited_list():
  ctx = _remember_ctx(ToolConfirmation(
      confirmed=True, payload={"facts": ["Lives in Porto"]}))
  out = await memory_tools.remember_facts(["Lives in Lisbon", "Likes Rust"], ctx)
  assert out["facts"] == ["Lives in Porto"]
  assert _saved_texts(ctx) == ["Lives in Porto"]


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", [
    ToolConfirmation(confirmed=False, payload={"facts": ["Likes Rust"]}),
    ToolConfirmation(confirmed=True, payload={"facts": []}),
])
async def test_remember_facts_declined_saves_nothing(confirmation):
  ctx = _remember_ctx(confirmation)
  out = await memory_tools.remember_facts(["Likes Rust"], ctx)
  assert out["status"] == "declined"
  ctx.add_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_remember_facts_reports_memory_errors():
  ctx = _remember_ctx(ToolConfirmation(confirmed=True))
  ctx.add_memory.side_effect = NotImplementedError("no direct writes")
  out = await memory_tools.remember_facts(["Likes Rust"], ctx)
  assert out["status"] == "error" and "no direct writes" in out["error_message"]
  assert (await memory_tools.remember_facts([], _remember_ctx()))["status"] == "error"
