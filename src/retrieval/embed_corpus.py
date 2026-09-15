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


def main():
    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    model = SentenceTransformer(
        MODEL_NAME
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    print(
        f"Embedded {len(chunks)} corpus chunks"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Saved to: {EMBEDDINGS_PATH}"
    )


if __name__ == "__main__":
    main()