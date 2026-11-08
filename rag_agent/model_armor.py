"""Screens user prompts and model replies with Google Cloud Model Armor.

* ``screen_prompt`` (before_model_callback) checks the user's new message
  before Gemini sees it. If a filter matches, Gemini is never called and the
  user gets a short refusal instead.
* ``screen_response`` (after_model_callback) checks each final text reply
  before it is shown, and replaces a reply that matches a filter.

Set ``MODEL_ARMOR_TEMPLATE`` to the template's full resource name to turn this
on; without it both callbacks do nothing. If Model Armor can't be reached the
message is blocked, unless ``MODEL_ARMOR_FAIL_OPEN=true``.
"""

import asyncio
import functools
import logging
import os
import re

import google.auth
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.auth.transport.requests import AuthorizedSession
from google.genai import types

logger = logging.getLogger(__name__)

_TEMPLATE_PATTERN = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)"
    r"/templates/[^/]+$")
_TIMEOUT_SECONDS = 10

PROMPT_BLOCKED_MESSAGE = (
    "I can't help with that request: it was blocked by the safety filter ({})."
)
RESPONSE_BLOCKED_MESSAGE = (
    "I can't share that answer: it was blocked by the safety filter ({})."
)


class ModelArmorError(RuntimeError):
  """Model Armor couldn't screen the text."""


def _template() -> str:
  return os.getenv("MODEL_ARMOR_TEMPLATE", "").strip()


def _fail_open() -> bool:
  return os.getenv("MODEL_ARMOR_FAIL_OPEN", "false").strip().lower() in (
      "1", "true")


@functools.lru_cache(maxsize=1)
def _session() -> AuthorizedSession:
  credentials, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"])
  return AuthorizedSession(credentials)


def matched_filters(result: dict) -> list[str]:
  """Returns the names of the filters that matched in a sanitizationResult."""
  if result.get("filterMatchState") != "MATCH_FOUND":
    return []
  matches = []
  for name, wrapper in (result.get("filterResults") or {}).items():
    for filter_result in (wrapper or {}).values():
      if not isinstance(filter_result, dict):
        continue
      # The sensitive-data filter nests its result one level deeper.
      inner = filter_result.get("inspectResult", filter_result)
      if inner.get("matchState") != "MATCH_FOUND":
        continue
      if name == "rai":
        types_ = [t for t, r in (inner.get("raiFilterTypeResults") or {}).items()
                  if r.get("matchState") == "MATCH_FOUND"]
        matches += [f"rai:{t}" for t in types_] or ["rai"]
      elif name == "sdp":
        info_types = sorted({f.get("infoType", "") for f in
                             inner.get("findings") or []} - {""})
        matches += [f"sdp:{t}" for t in info_types] or ["sdp"]
      else:
        matches.append(name)
  return matches or ["unknown_filter"]


def _sanitize_sync(kind: str, text: str) -> list[str]:
  template = _template()
  match = _TEMPLATE_PATTERN.match(template)
  if not match:
    raise ModelArmorError(
        "MODEL_ARMOR_TEMPLATE must look like"
        " projects/PROJECT/locations/LOCATION/templates/TEMPLATE_ID")
  location = match["location"]
  if kind == "prompt":
    method, body = "sanitizeUserPrompt", {"userPromptData": {"text": text}}
  else:
    method, body = "sanitizeModelResponse", {"modelResponseData": {"text": text}}
  url = (f"https://modelarmor.{location}.rep.googleapis.com/v1/"
         f"{template}:{method}")
  try:
    response = _session().post(
        url, json=body, timeout=_TIMEOUT_SECONDS,
        headers={"x-goog-user-project": match["project"]})
    response.raise_for_status()
    result = response.json().get("sanitizationResult", {})
  except Exception as e:  # pylint: disable=broad-except
    raise ModelArmorError(f"{type(e).__name__}: {e}") from e
  return matched_filters(result)


async def sanitize(kind: str, text: str) -> list[str]:
  """Returns matched filter names for ``text`` ("prompt" or "response")."""
  return await asyncio.to_thread(_sanitize_sync, kind, text)


async def _check(kind: str, text: str) -> list[str]:
  """Like sanitize, but applies the fail-open/fail-closed setting."""
  try:
    return await sanitize(kind, text)
  except ModelArmorError as e:
    if _fail_open():
      logger.warning("Model Armor unavailable, allowing %s: %s", kind, e)
      return []
    logger.error("Model Armor unavailable, blocking %s: %s", kind, e)
    return ["model_armor_unavailable"]


def _text_of(content: types.Content | None) -> str:
  if not content or not content.parts:
    return ""
  return "\n".join(p.text for p in content.parts
                   if p.text and not getattr(p, "thought", False)).strip()


def _reply(text: str) -> LlmResponse:
  return LlmResponse(content=types.Content(
      role="model", parts=[types.Part(text=text)]))


async def screen_prompt(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
  """Blocks the model call if the user's new message matches a filter."""
  del callback_context  # Unused.
  if not _template() or not llm_request.contents:
    return None
  last = llm_request.contents[-1]
  # Only a fresh user message: later model calls in the same turn end with a
  # tool result, which isn't the user's prompt.
  if last.role != "user" or any(p.function_response for p in last.parts or []):
    return None
  text = _text_of(last)
  if not text:
    return None
  matches = await _check("prompt", text)
  if not matches:
    return None
  logger.warning("Model Armor blocked a user prompt: %s", matches)
  return _reply(PROMPT_BLOCKED_MESSAGE.format(", ".join(matches)))


async def screen_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
  """Replaces a final text reply that matches a filter."""
  del callback_context  # Unused.
  if not _template() or llm_response.partial or not llm_response.content:
    return None
  if any(p.function_call for p in llm_response.content.parts or []):
    return None
  text = _text_of(llm_response.content)
  if not text:
    return None
  matches = await _check("response", text)
  if not matches:
    return None
  logger.warning("Model Armor blocked a model response: %s", matches)
  return _reply(RESPONSE_BLOCKED_MESSAGE.format(", ".join(matches)))
