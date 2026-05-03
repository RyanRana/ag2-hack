"""Build AG2 llm_config from environment variables.

LLM-backed agents (Skeptic §8.3, VLMReasoner §2.1) require an API key. We
do NOT silently degrade to a heuristic — per project memory, the AG2
``AssistantAgent + register_function`` idiom is load-bearing and must use a
real model. Tests inject a fake at construction time.

Reads ``.env`` from the project root automatically so users can drop their
key there without exporting it manually.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional but in pyproject deps
    pass


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class LLMKeyMissingError(RuntimeError):
    """Raised when an LLM-backed agent is constructed without a configured key."""


def llm_key_available() -> bool:
    """True if any supported LLM provider key is configured."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def openai_llm_config(*, model: str | None = None, temperature: float = 0.0) -> dict:
    """Return an AG2 llm_config dict for whichever provider has a key set.

    Anthropic is preferred when ``ANTHROPIC_API_KEY`` is set; otherwise
    OpenAI is used. Pulse's Skeptic + VLMReasoner work identically against
    either provider — AG2 abstracts the chat API.

    The function name is kept for backwards compatibility; new code should
    call :func:`build_llm_config`.
    """
    return build_llm_config(model=model, temperature=temperature)


def build_llm_config(*, model: str | None = None, temperature: float = 0.0) -> dict:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    override_model = model or os.environ.get("PULSE_LLM_MODEL")
    # Supports OpenRouter (and other OpenAI-compatible gateways).
    openai_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
        "OPENROUTER_BASE_URL"
    )

    if anthropic_key:
        chosen_model = override_model or DEFAULT_ANTHROPIC_MODEL
        return {
            "config_list": [
                {
                    "model": chosen_model,
                    "api_key": anthropic_key,
                    "api_type": "anthropic",
                }
            ],
            "temperature": temperature,
            "cache_seed": None,
        }
    if openai_key:
        chosen_model = override_model or DEFAULT_OPENAI_MODEL
        openai_config: dict[str, str] = {
            "model": chosen_model,
            "api_key": openai_key,
        }
        if openai_base_url:
            openai_config["base_url"] = openai_base_url
        return {
            "config_list": [
                openai_config
            ],
            "temperature": temperature,
            "cache_seed": None,
        }
    raise LLMKeyMissingError(
        "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set. Pulse's "
        "Skeptic and VLMReasoner agents use a real LLM via AG2 AssistantAgent. "
        "Drop a key in .env or export it before running these agents."
    )
