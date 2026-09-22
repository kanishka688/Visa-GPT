import json
import random
import time
from typing import Optional

import requests

from openai import (
    OpenAI,
    RateLimitError,
)

from src.core.settings import (
    LLM_PROVIDER,
    LLM_MODEL,
    OLLAMA_URL,
    OPENAI_MODEL,
)


# ==================================================
# OPENAI CONFIG
# ==================================================

OPENAI_MAX_RETRIES = 5

OPENAI_CLIENT = OpenAI(
    # Disable SDK retries so our benchmark retry
    # behavior is explicit and measurable.
    max_retries=0,
    timeout=120.0,
)


# ==================================================
# RATE-LIMIT HELPERS
# ==================================================


def is_non_retryable_rate_limit(
    exc: RateLimitError,
) -> bool:
    """
    Detect 429 errors that retries cannot fix.

    Examples:
    - no credits
    - insufficient quota
    - spend limit reached
    """

    message = str(exc).lower()

    non_retryable_markers = (
        "credit_balance_exhausted",
        "insufficient_quota",
        "organization_usage_limit_exceeded",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "no credits remaining",
    )

    return any(
        marker in message
        for marker in non_retryable_markers
    )


def get_backoff_seconds(
    retry_number: int,
) -> float:
    """
    Exponential backoff with jitter.

    Retry 1: ~2 sec
    Retry 2: ~4 sec
    Retry 3: ~8 sec
    Retry 4: ~16 sec
    Retry 5: ~30 sec
    """

    base_delay = min(
        2 ** retry_number,
        30,
    )

    jitter = random.uniform(
        0.0,
        1.0,
    )

    return (
        base_delay
        + jitter
    )


def execute_openai_request(
    request_callable,
):
    """
    Execute an OpenAI request with bounded
    exponential backoff for temporary rate limits.
    """

    for attempt in range(
        OPENAI_MAX_RETRIES + 1
    ):

        try:

            return request_callable()

        except RateLimitError as exc:

            if is_non_retryable_rate_limit(
                exc
            ):
                raise

            if attempt >= OPENAI_MAX_RETRIES:
                raise

            retry_number = (
                attempt + 1
            )

            delay = get_backoff_seconds(
                retry_number
            )

            print(
                "[OpenAI rate limit] "
                f"retry {retry_number}/"
                f"{OPENAI_MAX_RETRIES} "
                f"in {delay:.1f}s"
            )

            time.sleep(
                delay
            )

    raise RuntimeError(
        "OpenAI retry loop exited unexpectedly."
    )


# ==================================================
# OPENAI — TEXT
# ==================================================


def call_openai(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 500,
) -> str:
    """
    Call OpenAI for normal text generation.
    """

    def make_request():

        return OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,

            reasoning={
                "effort": "low",
            },

            max_output_tokens=(
                max_output_tokens
            ),

            store=False,
        )

    response = execute_openai_request(
        make_request
    )

    output_text = (
        response.output_text
        or ""
    ).strip()

    if not output_text:

        raise ValueError(
            "OpenAI returned empty text output."
        )

    return output_text


# ==================================================
# OPENAI — STRUCTURED JSON
# ==================================================


def call_openai_json(
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict,
    max_output_tokens: int = 500,
) -> dict:
    """
    Call OpenAI using Structured Outputs.

    The model must return JSON matching the
    supplied schema.
    """

    def make_request():

        return OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,

            reasoning={
                "effort": "low",
            },

            max_output_tokens=(
                max_output_tokens
            ),

            store=False,

            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )

    response = execute_openai_request(
        make_request
    )

    raw = (
        response.output_text
        or ""
    ).strip()

    if not raw:

        raise ValueError(
            "OpenAI returned empty "
            "structured output."
        )

    return json.loads(
        raw
    )


# ==================================================
# OLLAMA — TEXT
# ==================================================


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """
    Call the local Ollama model.
    """

    payload = {
        "model": LLM_MODEL,

        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "stream": False,

        "options": {
            "temperature": temperature,
        },
    }

    if json_mode:

        payload[
            "format"
        ] = "json"

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return (
        data["message"]["content"]
        .strip()
    )


# ==================================================
# OLLAMA — JSON
# ==================================================


def call_ollama_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> dict:
    """
    Call Ollama in JSON mode.
    """

    raw = call_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        json_mode=True,
    )

    return json.loads(
        raw
    )


# ==================================================
# PROVIDER — TEXT
# ==================================================


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 500,
    provider: Optional[str] = None,
) -> str:
    """
    Provider-independent text generation.

    Provider comes from:

    KKGPT_LLM_PROVIDER=openai

    or:

    KKGPT_LLM_PROVIDER=ollama
    """

    selected_provider = (
        provider
        or LLM_PROVIDER
    ).lower()

    if selected_provider == "openai":

        return call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

    if selected_provider == "ollama":

        return call_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: "
        f"{selected_provider}"
    )


# ==================================================
# PROVIDER — STRUCTURED JSON
# ==================================================


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict,
    temperature: float = 0.0,
    max_output_tokens: int = 500,
    provider: Optional[str] = None,
) -> dict:
    """
    Provider-independent structured generation.

    OpenAI:
        strict JSON-schema Structured Outputs

    Ollama:
        JSON mode
    """

    selected_provider = (
        provider
        or LLM_PROVIDER
    ).lower()

    if selected_provider == "openai":

        return call_openai_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
            schema=schema,
            max_output_tokens=max_output_tokens,
        )

    if selected_provider == "ollama":

        return call_ollama_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: "
        f"{selected_provider}"
    )