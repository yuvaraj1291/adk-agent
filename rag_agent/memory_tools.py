"""Saving to long-term memory only after the user confirms.

The agent calls ``remember_facts`` with the facts it wants to keep. The first
call doesn't save anything: it asks the user to confirm and shows them the
facts. In ``adk web`` this appears as a confirmation card with the facts as
editable JSON, so the user can drop or reword facts before approving. When the
user answers, ADK runs the tool again with their answer, and only an approved
list is written to the memory service.
"""

from typing import Any

from google.adk.memory.memory_entry import MemoryEntry
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MAX_FACTS = 10
MAX_FACT_LENGTH = 300


def _clean_facts(facts: Any) -> list[str]:
  if isinstance(facts, str):
    facts = [facts]
  if not isinstance(facts, list):
    return []
  cleaned = []
  for fact in facts:
    if isinstance(fact, str) and fact.strip():
      text = " ".join(fact.split())[:MAX_FACT_LENGTH]
      if text not in cleaned:
        cleaned.append(text)
  return cleaned[:MAX_FACTS]


def _confirmation_hint(facts: list[str]) -> str:
  lines = "\n".join(f"- {fact}" for fact in facts)
  return (
      "Save these to long-term memory?\n"
      f"{lines}\n"
      "You can edit the list before confirming, or decline to save nothing."
  )


async def remember_facts(facts: list[str], tool_context: ToolContext) -> dict:
  """Saves facts about the user to long-term memory, after the user confirms.

  Call this when the user shares a lasting fact or preference about
  themselves, or asks you to remember something. Nothing is saved until the
  user approves; they see the exact facts first and can edit them.

  Args:
    facts: Short, self-contained facts in the third person, for example
      "Prefers answers in bullet points". At most 10.

  Returns:
    A dict with ``status``: "awaiting_confirmation" while the user decides,
    then "saved" (with the facts that were saved), "declined" or "error".
  """
  proposed = _clean_facts(facts)
  if not proposed:
    return {"status": "error",
            "error_message": "No facts to save. Pass a list of short facts."}

  confirmation = tool_context.tool_confirmation
  if confirmation is None:
    tool_context.request_confirmation(
        hint=_confirmation_hint(proposed),
        payload={"facts": proposed},
    )
    return {"status": "awaiting_confirmation", "facts": proposed}

  if not confirmation.confirmed:
    return {"status": "declined",
            "message": "The user chose not to save these. Nothing was saved."}

  # The user may have edited the list in the confirmation card.
  approved = proposed
  payload = confirmation.payload
  if isinstance(payload, dict) and "facts" in payload:
    approved = _clean_facts(payload["facts"])
  if not approved:
    return {"status": "declined",
            "message": "The user removed every fact. Nothing was saved."}

  try:
    await tool_context.add_memory(
        memories=[
            MemoryEntry(content=types.Content(
                role="user", parts=[types.Part(text=fact)]))
            for fact in approved
        ],
        # Merge with existing memories instead of adding duplicates.
        custom_metadata={"enable_consolidation": True},
    )
  except (NotImplementedError, ValueError) as e:
    return {"status": "error",
            "error_message": f"Memory isn't available here: {e}"}
  except Exception as e:  # pylint: disable=broad-except
    return {"status": "error",
            "error_message": f"Saving to memory failed: {type(e).__name__}: {e}"}
  return {"status": "saved", "facts": approved}
