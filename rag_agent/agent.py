"""A simple ADK agent with Google Search and Vertex AI Vector Search (RAG)."""

import logging
import os
import pathlib

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from .vector_search import search_knowledge_base

load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)

logger = logging.getLogger(__name__)


async def save_session_to_memory(callback_context: CallbackContext) -> None:
  """Sends only this turn's events to the memory service after each turn.

  Earlier turns were already sent, so resending the whole session would make
  Memory Bank re-read the same messages every turn. Every turn of a session is
  sent on the same ``stream_id`` (the session ID), so Memory Bank still treats
  them as one conversation. A memory failure is logged and never breaks the
  reply.
  """
  try:
    session = callback_context.session
    turn_events = [
        e for e in session.events
        if e.invocation_id == callback_context.invocation_id
    ]
    if not turn_events:
      return
    logger.info("Saving %d of %d session events to memory",
                len(turn_events), len(session.events))
    await callback_context.add_events_to_memory(
        events=turn_events,
        custom_metadata={"stream_id": session.id},
    )
  except Exception:  # pylint: disable=broad-except
    logger.exception("Failed to save turn to memory")

root_agent = Agent(
    name="rag_search_agent",
    model=os.getenv("AGENT_MODEL", "gemini-3.6-flash"),
    description=(
        "Answers questions using a private knowledge base (Vector Search) and"
        " Google Search."
    ),
    instruction="""You are a helpful research assistant with two tools:

1. `search_knowledge_base` - searches the private knowledge base (internal
   documents). Use it FIRST for anything that might be covered by internal or
   company-specific documents.
2. `google_search` - searches the public web. Use it for current events,
   general/public knowledge, or when the knowledge base has no relevant results.

Rules:
- Base your answers on tool results; do not invent facts.
- When you use the knowledge base, cite the `source` of the passages you used.
- When you use Google Search, mention that the information came from the web.
- If neither tool gives an answer, say you don't know.

Memory: facts from your earlier conversations with this user may be included
below. Use them to personalise answers (for example preferences or details the
user told you), but don't treat them as a source for document facts.
""",
    tools=[
        # Gemini built-in search can't be mixed with function tools in one
        # request; bypass_multi_tools_limit makes ADK wrap it in a sub-agent.
        GoogleSearchTool(bypass_multi_tools_limit=True),
        search_knowledge_base,
        # Searches the memory service with the user's message and adds any
        # matching memories to the prompt before each model call.
        PreloadMemoryTool(),
    ],
    after_agent_callback=save_session_to_memory,
)
