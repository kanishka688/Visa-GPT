from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path(
    "data/processed/uscis_stem_opt_policy_alert_chunks.json"
)

EMBEDDINGS_PATH = Path(
    "data/processed/uscis_stem_opt_embeddings.npy"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Load our 6 chunks
    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    # Extract only the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Load the local embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Convert each chunk into a numeric vector
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save vectors locally
    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    print(f"Embedded {len(chunks)} chunks")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()