from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path(
    "data/processed/corpus_chunks.json"
)

EMBEDDINGS_PATH = Path(
    "data/processed/corpus_embeddings.npy"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def main():
    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )

    document_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    question = input(
        "Ask KK-Gpt: "
    ).strip()

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = (
        document_embeddings
        @ query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:TOP_K]

    print("\nTop results:\n")

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        chunk = chunks[index]

        print(
            f"--- RESULT {rank} ---"
        )

        print(
            f"Score: "
            f"{scores[index]:.4f}"
        )

        print(
            f"Agency: "
            f"{chunk['agency']}"
        )

        print(
            f"Document: "
            f"{chunk['document']}"
        )

        print(
            f"Topic: "
            f"{chunk['topic']}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        print()

        print(
            chunk["text"]
        )

        print("\n")


if __name__ == "__main__":
    main()