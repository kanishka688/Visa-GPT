import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def get_positive_int(
    name: str,
    default: int,
) -> int:

    raw_value = os.getenv(
        name,
        str(default),
    )

    try:
        value = int(raw_value)

    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0."
        )

    return value


def get_llm_provider() -> str:

    provider = os.getenv(
        "KKGPT_LLM_PROVIDER",
        "ollama",
    ).strip().lower()

    allowed = {
        "ollama",
        "openai",
    }

    if provider not in allowed:
        raise ValueError(
            "KKGPT_LLM_PROVIDER must be "
            "'ollama' or 'openai'."
        )

    return provider


LLM_PROVIDER = get_llm_provider()

OPENAI_MODEL = os.getenv(
    "KKGPT_OPENAI_MODEL",
    "gpt-5.6-luna",
)

LLM_MODEL = os.getenv(
    "KKGPT_LLM_MODEL",
    "qwen3:4b-instruct",
)

OLLAMA_URL = os.getenv(
    "KKGPT_OLLAMA_URL",
    "http://localhost:11434/api/chat",
)

RETRIEVAL_TOP_K = get_positive_int(
    "KKGPT_RETRIEVAL_TOP_K",
    5,
)

MAX_EVIDENCE_SOURCES = get_positive_int(
    "KKGPT_MAX_EVIDENCE_SOURCES",
    5,
)