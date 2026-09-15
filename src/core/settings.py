import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


def get_positive_int(
    name: str,
    default: int,
) -> int:

    raw_value = os.getenv(
        name,
        str(default),
    )

    try:
        value = int(
            raw_value
        )

    except ValueError as exc:

        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if value <= 0:

        raise ValueError(
            f"{name} must be greater than 0."
        )

    return value


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