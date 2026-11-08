"""A simple ADK agent with Google Search and Vertex AI Vector Search (RAG)."""

import os
import pathlib

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

from .memory_tools import remember_facts
from .model_armor import screen_prompt, screen_response
from .search_budget import drop_spent_tools, limit_searches
from .vector_search import search_knowledge_base

load_dotenv(pathlib.Path(__file__).parent / ".env", override=True)

root_agent = Agent(
    name="rag_search_agent",
    model=os.getenv("AGENT_MODEL", "gemini-3.6-flash"),
    description=(
        "Answers questions using a private knowledge base (Vector Search) and"
        " Google Search."
    ),
    instruction="""You are a helpful research assistant with these tools:

1. `search_knowledge_base` - searches the private knowledge base (internal
   documents). Use it FIRST for anything that might be covered by internal or
   company-specific documents.
2. `google_search` - searches the public web. Use it for current events,
   general/public knowledge, or when the knowledge base has no relevant results.
3. `remember_facts` - proposes facts about the user to save to long-term memory.

Rules:
- Base your answers on tool results; do not invent facts.
- When you use the knowledge base, cite the `source` of the passages you used.
- When you use Google Search, mention that the information came from the web.
- If neither tool gives an answer, say you don't know.
- Each question has a small search budget for each tool. Make every search
  count, and don't repeat a search you've already run.

Memory:
- Facts from your earlier conversations with this user may be included below.
  Use them to personalise answers (for example preferences or details the user
  told you), but don't treat them as a source for document facts.
- Nothing is saved to memory unless you call `remember_facts` and the user
  approves. Call it when the user shares a lasting fact or preference about
  themselves, or asks you to remember something. Pass short third-person facts
  ("Prefers answers in bullet points"). Never propose document contents,
  passwords, financial or ID numbers, or other secrets.
- The user is shown the exact facts and asked to confirm. Only say something
  was saved when the result status is "saved", and mention the saved facts. If
  they decline, don't propose the same facts again.
""",
    tools=[
        # Gemini built-in search can't be mixed with function tools in one
        # request; bypass_multi_tools_limit makes ADK wrap it in a sub-agent.
        GoogleSearchTool(bypass_multi_tools_limit=True),
        search_knowledge_base,
        # Saves facts only after the user confirms them (see memory_tools.py).
        remember_facts,
        # Searches the memory service with the user's message and adds any
        # matching memories to the prompt before each model call.
        PreloadMemoryTool(),
    ],
    # Each turn gets a small search budget; once it is spent the tools are
    # removed so the model has to answer (see search_budget.py).
    before_tool_callback=limit_searches,
    # Model Armor screens the user's message first; a blocked prompt never
    # reaches Gemini (see model_armor.py).
    before_model_callback=[screen_prompt, drop_spent_tools],
    after_model_callback=screen_response,
)
