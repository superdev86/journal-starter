"""Task 4: Implement analyze_journal_entry using any OpenAI-compatible API.

This project mandates the OpenAI Python SDK, which works with:

- GitHub Models (default, free, no credit card required)
- OpenAI proper
- Azure OpenAI
- Groq, Together, OpenRouter, Fireworks, DeepInfra
- Ollama, LM Studio, vLLM (local)
- Anthropic via their OpenAI-compat endpoint

Set OPENAI_API_KEY, and optionally OPENAI_BASE_URL and OPENAI_MODEL
in your .env file. Settings are loaded by `api.config.Settings`.
"""

import json
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from api.config import get_settings


def _default_client() -> AsyncOpenAI:
    """Construct the real OpenAI client from application settings."""
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Analyze a journal entry using an OpenAI-compatible LLM."""
    if client is None:
        client = _default_client()

    settings = get_settings()

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You analyze journal entries. "
                "Return ONLY valid JSON with exactly these fields: "
                "sentiment, summary, topics. "
                "sentiment must be one of: positive, negative, neutral. "
                "summary must be a concise summary of the journal entry. "
                "topics must be a JSON array of strings."
            ),
        },
        {
            "role": "user",
            "content": f"Analyze this journal entry:\n\n{entry_text}",
        },
    ]

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned an empty response")

    analysis = cast(dict, json.loads(content))

    return {
        "entry_id": entry_id,
        "sentiment": analysis["sentiment"],
        "summary": analysis["summary"],
        "topics": analysis["topics"],
    }
