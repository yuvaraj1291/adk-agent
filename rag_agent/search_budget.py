"""Per-turn limits on how many searches the agent may run.

Without a limit the model can keep searching for as long as ADK allows
(500 model calls per turn by default). Here each turn gets a small budget:

* ``before_tool_callback`` (``limit_searches``) counts every search and refuses
  calls past the limit with a ``budget_exhausted`` result.
* ``before_model_callback`` (``drop_spent_tools``) removes a search tool from
  the request once its budget is spent, and tells the model to answer with
  what it already has. With no search tools left the model has to reply in
  text, which ends the turn.
"""

import os
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

KNOWLEDGE_BASE_TOOL = "search_knowledge_base"
# GoogleSearchTool(bypass_multi_tools_limit=True) is exposed to the model as a
# sub-agent tool with this name.
WEB_SEARCH_TOOL = "google_search_agent"

# Turn-scoped state: "temp:" keys are never persisted. Counts are also tagged
# with the invocation ID so a value left over from an earlier turn is ignored.
_STATE_KEY = "temp:search_calls"


def search_limits() -> dict[str, int]:
  """Returns the maximum number of calls per turn for each search tool."""
  return {
      KNOWLEDGE_BASE_TOOL: int(os.getenv("MAX_KNOWLEDGE_BASE_SEARCHES", "3")),
      WEB_SEARCH_TOOL: int(os.getenv("MAX_WEB_SEARCHES", "2")),
  }


def _turn_counts(context: CallbackContext) -> dict[str, int]:
  data = context.state.get(_STATE_KEY) or {}
  if data.get("invocation_id") != context.invocation_id:
    return {}
  return dict(data.get("counts", {}))


def limit_searches(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
  """Counts search calls and refuses any past this turn's limit."""
  del args  # Unused.
  limit = search_limits().get(tool.name)
  if limit is None:
    return None
  counts = _turn_counts(tool_context)
  used = counts.get(tool.name, 0)
  if used >= limit:
    return {
        "status": "budget_exhausted",
        "error_message": (
            f"Search limit reached: {tool.name} can be used {limit} times per"
            " question. Answer from the results you already have, or say you"
            " don't know."
        ),
    }
  counts[tool.name] = used + 1
  tool_context.state[_STATE_KEY] = {
      "invocation_id": tool_context.invocation_id,
      "counts": counts,
  }
  return None


def drop_spent_tools(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
  """Removes search tools whose budget is spent from the next model call."""
  counts = _turn_counts(callback_context)
  spent = {
      name for name, limit in search_limits().items()
      if counts.get(name, 0) >= limit
  }
  if not spent or not llm_request.config.tools:
    return None

  kept_tools = []
  for tool in llm_request.config.tools:
    declarations = getattr(tool, "function_declarations", None)
    if declarations:
      tool.function_declarations = [
          d for d in declarations if d.name not in spent
      ] or None
      if not tool.function_declarations and not _has_other_fields(tool):
        continue
    kept_tools.append(tool)
  llm_request.config.tools = kept_tools or None
  # tools_dict is left alone: a stray call to a removed tool still reaches
  # limit_searches and gets a budget_exhausted result instead of an error.

  llm_request.append_instructions([
      "You have used all your searches for "
      + " and ".join(sorted(spent))
      + " this turn. Do not call "
      + ("them" if len(spent) > 1 else "it")
      + " again. Answer from the results you already have, or say you don't"
      " know."
  ])
  return None


def _has_other_fields(tool: Any) -> bool:
  dump = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else {}
  dump.pop("function_declarations", None)
  return bool(dump)
