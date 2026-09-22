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

INPUT_FILE = Path(
    "data/social/anlf/anlf_relevant.jsonl"
)

INDEX_DIR = Path(
    "data/social/anlf/index"
)

CHUNKS_FILE = INDEX_DIR / "anlf_chunks.jsonl"

EMBEDDINGS_FILE = INDEX_DIR / "anlf_embeddings.npy"


# ==================================================
# LOAD RELEVANT POSTS
# ==================================================


def load_posts() -> list[dict]:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Relevant ANLF dataset not found: "
            f"{INPUT_FILE}"
        )

    posts = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:

                continue

            posts.append(
                json.loads(
                    line
                )
            )

    return posts


# ==================================================
# ADMIN TEXT
# ==================================================


def build_admin_text(
    post: dict,
) -> str:

    info = (
        post.get(
            "admin_info"
        )
        or {}
    )

    parts = []

    topic = post.get(
        "topic"
    )

    if topic:

        parts.append(
            f"Topic: {topic}"
        )

    content_type = post.get(
        "content_type"
    )

    if content_type:

        parts.append(
            f"Content type: {content_type}"
        )

    summary = info.get(
        "summary"
    )

    if summary:

        parts.append(
            f"Admin summary: {summary}"
        )

    caption = post.get(
        "caption"
    )

    if caption:

        parts.append(
            f"Post caption: {caption}"
        )

    fields = [
        (
            "Interview questions",
            "interview_questions",
        ),
        (
            "Experience notes",
            "experience_notes",
        ),
        (
            "Travel notes",
            "travel_notes",
        ),
        (
            "Stamping notes",
            "stamping_notes",
        ),
        (
            "Employment notes",
            "employment_notes",
        ),
        (
            "Admin tips",
            "admin_tips",
        ),
    ]

    for label, key in fields:

        values = info.get(
            key,
            [],
        )

        if values:

            formatted = "\n".join(
                f"- {item}"
                for item in values
                if item
            )

            if formatted:

                parts.append(
                    f"{label}:\n{formatted}"
                )

    return "\n\n".join(
        parts
    ).strip()


# ==================================================
# CHUNK CREATION
# ==================================================


def build_chunks(
    posts: list[dict],
) -> list[dict]:

    chunks = []

    for post in posts:

        post_id = post.get(
            "post_id"
        )

        if not post_id:

            continue

        common = {
            "post_id": post_id,

            "account": post.get(
                "account"
            ),

            "platform": post.get(
                "platform"
            ),

            "post_url": post.get(
                "post_url"
            ),

            "posted_at": post.get(
                "posted_at"
            ),

            "topic": post.get(
                "topic"
            ),

            "content_type": post.get(
                "content_type"
            ),

            "source_class": post.get(
                "source_class",
                "SOCIAL_DISCUSSION",
            ),

            "relevance_categories": (
                post.get(
                    "relevance",
                    {},
                ).get(
                    "categories",
                    [],
                )
            ),
        }

        # ------------------------------------------
        # ADMIN / POST INFORMATION
        # ------------------------------------------

        admin_text = (
            build_admin_text(
                post
            )
        )

        if admin_text:

            chunks.append(
                {
                    **common,

                    "chunk_id": (
                        f"{post_id}:admin"
                    ),

                    "chunk_type": (
                        "ADMIN_INFO"
                    ),

                    "text": (
                        admin_text
                    ),

                    "raw_text": (
                        post.get(
                            "caption",
                            "",
                        )
                    ),
                }
            )

        # ------------------------------------------
        # COMMENTS
        # ------------------------------------------

        admin_summary = (
            post.get(
                "admin_info",
                {},
            ).get(
                "summary",
                "",
            )
        )

        for comment in post.get(
            "comments",
            [],
        ):

            comment_id = (
                comment.get(
                    "comment_id"
                )
            )

            comment_text = (
                comment.get(
                    "text",
                    ""
                )
                .strip()
            )

            if (
                not comment_id
                or not comment_text
            ):

                continue

            contextual_text = (
                f"Topic: "
                f"{post.get('topic', '')}\n\n"
                f"Post context: "
                f"{admin_summary}\n\n"
                f"Community comment: "
                f"{comment_text}"
            ).strip()

            chunks.append(
                {
                    **common,

                    "chunk_id": (
                        f"{post_id}:comment:"
                        f"{comment_id}"
                    ),

                    "chunk_type": (
                        "COMMENT"
                    ),

                    "comment_id": (
                        comment_id
                    ),

                    "comment_posted_at": (
                        comment.get(
                            "posted_at"
                        )
                    ),

                    "text": (
                        contextual_text
                    ),

                    "raw_text": (
                        comment_text
                    ),
                }
            )

    return chunks


# ==================================================
# SAVE INDEX
# ==================================================


def save_chunks(
    chunks: list[dict],
):

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CHUNKS_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for chunk in chunks:

            handle.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )


def build_embeddings(
    chunks: list[dict],
):

    if not chunks:

        raise RuntimeError(
            "No ANLF chunks were created."
        )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    return embeddings


# ==================================================
# CLI
# ==================================================


def main():

    print(
        "Building ANLF social index..."
    )

    posts = load_posts()

    chunks = build_chunks(
        posts
    )

    save_chunks(
        chunks
    )

    embeddings = (
        build_embeddings(
            chunks
        )
    )

    admin_chunks = sum(
        1
        for chunk in chunks
        if chunk["chunk_type"]
        == "ADMIN_INFO"
    )

    comment_chunks = sum(
        1
        for chunk in chunks
        if chunk["chunk_type"]
        == "COMMENT"
    )

    print()

    print(
        "=" * 60
    )

    print(
        "ANLF INDEX COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Relevant posts:    {len(posts)}"
    )

    print(
        f"Admin chunks:      {admin_chunks}"
    )

    print(
        f"Comment chunks:    {comment_chunks}"
    )

    print(
        f"Total chunks:      {len(chunks)}"
    )

    print(
        f"Embedding shape:   "
        f"{embeddings.shape}"
    )

    print(
        f"Chunks file:       {CHUNKS_FILE}"
    )

    print(
        f"Embeddings file:   "
        f"{EMBEDDINGS_FILE}"
    )


if __name__ == "__main__":

    main()