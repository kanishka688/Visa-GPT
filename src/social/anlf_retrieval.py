import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.search_with_reranker import (
    EMBEDDING_MODEL_NAME,
)


# ==================================================
# CONFIG
# ==================================================

INDEX_DIR = Path(
    "data/social/anlf/index"
)

CHUNKS_FILE = (
    INDEX_DIR
    / "anlf_chunks.jsonl"
)

EMBEDDINGS_FILE = (
    INDEX_DIR
    / "anlf_embeddings.npy"
)

DEFAULT_TOP_K = 5

MAX_RESULTS_PER_POST = 3

MIN_SIMILARITY_SCORE = 0.30

MAX_SCORE_DROP_FROM_BEST = 0.12


# ==================================================
# LOAD INDEX
# ==================================================


def load_chunks() -> list[dict]:

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"ANLF chunks not found: "
            f"{CHUNKS_FILE}"
        )

    chunks = []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = (
                line.strip()
            )

            if not line:

                continue

            chunks.append(
                json.loads(
                    line
                )
            )

    return chunks


def load_index():

    chunks = (
        load_chunks()
    )

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"ANLF embeddings not found: "
            f"{EMBEDDINGS_FILE}"
        )

    embeddings = (
        np.load(
            EMBEDDINGS_FILE
        )
    )

    if len(
        chunks
    ) != len(
        embeddings
    ):

        raise RuntimeError(
            "ANLF chunks and embeddings "
            "are out of sync."
        )

    model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    return (
        chunks,
        embeddings,
        model,
    )


# ==================================================
# RETRIEVAL
# ==================================================


def retrieve_anlf(
    question: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:

    cleaned_question = (
        question.strip()
    )

    if not cleaned_question:

        return []

    query_embedding = (
        embedding_model.encode(
            [
                cleaned_question
            ],
            normalize_embeddings=True,
        )[0]
    )

    scores = (
        embeddings
        @ query_embedding
    )

    ranked_indices = (
        np.argsort(
            scores
        )[::-1]
    )

    if len(
        ranked_indices
    ) == 0:

        return []

    best_score = float(
        scores[
            int(
                ranked_indices[0]
            )
        ]
    )

    dynamic_threshold = max(
        MIN_SIMILARITY_SCORE,
        (
            best_score
            - MAX_SCORE_DROP_FROM_BEST
        ),
    )

    results = []

    per_post_counts = {}

    for index in ranked_indices:

        index = int(
            index
        )

        score = float(
            scores[
                index
            ]
        )

        if (
            score
            < dynamic_threshold
        ):

            break

        chunk = (
            chunks[
                index
            ]
        )

        post_id = (
            chunk.get(
                "post_id"
            )
        )

        count = (
            per_post_counts.get(
                post_id,
                0,
            )
        )

        if (
            count
            >= MAX_RESULTS_PER_POST
        ):

            continue

        result = {
            **chunk,

            "score": (
                score
            ),
        }

        results.append(
            result
        )

        per_post_counts[
            post_id
        ] = (
            count + 1
        )

        if len(
            results
        ) >= top_k:

            break

    return results


# ==================================================
# CLI
# ==================================================


def main():

    print(
        "Loading ANLF social index..."
    )

    (
        chunks,
        embeddings,
        model,
    ) = load_index()

    question = input(
        "ANLF question: "
    ).strip()

    results = (
        retrieve_anlf(
            question=(
                question
            ),

            chunks=(
                chunks
            ),

            embeddings=(
                embeddings
            ),

            embedding_model=(
                model
            ),
        )
    )

    print()

    print(
        "=" * 60
    )

    print(
        "ANLF RETRIEVAL RESULTS"
    )

    print(
        "=" * 60
    )

    if not results:

        print()

        print(
            "No sufficiently relevant "
            "ANLF content found."
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            f"[{index}] "
            f"{result['chunk_type']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Topic: "
            f"{result.get('topic')}"
        )

        print(
            f"URL: "
            f"{result.get('post_url')}"
        )

        print()

        print(
            result.get(
                "raw_text",
                "",
            )
        )


if __name__ == "__main__":

    main()