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

TOP_K = 3


def main():
    # Load chunks
    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    # Load document embeddings we created earlier
    document_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    # Load the same embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Ask user for a question
    question = input("Ask Visa-GPT: ").strip()

    # Convert question into an embedding
    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Compare question against every document chunk
    scores = document_embeddings @ query_embedding

    # Sort highest similarity first
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    print("\nTop results:\n")

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        print(f"--- RESULT {rank} ---")
        print(f"Score: {scores[index]:.4f}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print()
        print(chunk["text"])
        print("\n")


if __name__ == "__main__":
    main()