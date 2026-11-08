"""A simple ADK agent with Google Search and Vertex AI Vector Search (RAG)."""

import os
import pathlib

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool

from .vector_search import search_knowledge_base

load_dotenv(pathlib.Path(__file__).parent / ".env")

root_agent = Agent(
    name="rag_search_agent",
    model=os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
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
""",
    tools=[
        # Gemini built-in search can't be mixed with function tools in one
        # request; bypass_multi_tools_limit makes ADK wrap it in a sub-agent.
        GoogleSearchTool(bypass_multi_tools_limit=True),
        search_knowledge_base,
    ],
)
