from pathlib import Path
import json
import re


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/corpus_chunks.json"
)


# ==================================================
# CHUNK SETTINGS
# ==================================================

MAX_CHARS = 1200

MIN_CHARS = 80


# ==================================================
# TEXT HELPERS
# ==================================================


def normalize_text(
    text,
):
    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def split_sentences(
    text,
):
    """
    Lightweight sentence splitting.

    We avoid adding another NLP dependency.
    """

    text = normalize_text(
        text
    )

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ==================================================
# PARAGRAPH SPLITTING
# ==================================================


def split_into_paragraphs(
    text,
):
    text = normalize_text(
        text
    )

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


# ==================================================
# LONG PARAGRAPH HANDLING
# ==================================================


def split_long_paragraph(
    paragraph,
):
    """
    Split paragraphs larger than MAX_CHARS using
    sentence boundaries.
    """

    if len(
        paragraph
    ) <= MAX_CHARS:

        return [
            paragraph
        ]

    sentences = split_sentences(
        paragraph
    )

    chunks = []

    current = ""

    for sentence in sentences:

        candidate = (
            f"{current} {sentence}"
            if current
            else sentence
        )

        if (
            len(candidate)
            <= MAX_CHARS
        ):

            current = candidate

            continue

        if current:

            chunks.append(
                current.strip()
            )

        # Extremely long single sentence.
        if len(
            sentence
        ) > MAX_CHARS:

            for start in range(
                0,
                len(sentence),
                MAX_CHARS,
            ):

                piece = sentence[
                    start:
                    start + MAX_CHARS
                ].strip()

                if piece:
                    chunks.append(
                        piece
                    )

            current = ""

        else:

            current = sentence

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ==================================================
# DOCUMENT CHUNKING
# ==================================================


def chunk_document(
    document,
):
    """
    Convert one raw document into retrieval chunks.

    All source metadata is copied into every chunk.
    """

    metadata = document.get(
        "metadata",
        {},
    )

    text = document.get(
        "text",
        "",
    )

    paragraphs = split_into_paragraphs(
        text
    )

    pieces = []

    for paragraph in paragraphs:

        pieces.extend(
            split_long_paragraph(
                paragraph
            )
        )

    chunks = []

    current = ""

    for piece in pieces:

        candidate = (
            f"{current}\n\n{piece}"
            if current
            else piece
        )

        if (
            len(candidate)
            <= MAX_CHARS
        ):

            current = candidate

            continue

        if (
            current
            and len(current) >= MIN_CHARS
        ):

            chunks.append(
                current.strip()
            )

        current = piece

    if (
        current
        and len(current) >= MIN_CHARS
    ):

        chunks.append(
            current.strip()
        )

    return build_chunk_records(
        metadata,
        chunks,
    )


# ==================================================
# CHUNK RECORDS
# ==================================================


def build_chunk_records(
    metadata,
    texts,
):
    source_id = metadata.get(
        "id",
        "unknown",
    )

    records = []

    for index, text in enumerate(
        texts
    ):

        record = {
            # --------------------------------------
            # CHUNK IDENTITY
            # --------------------------------------

            "chunk_id": (
                f"{source_id}_chunk_{index:04d}"
            ),

            "source_id": (
                source_id
            ),

            # --------------------------------------
            # EXISTING RETRIEVAL METADATA
            # --------------------------------------

            "agency": metadata.get(
                "agency"
            ),

            "document": metadata.get(
                "title"
            ),

            "stage": metadata.get(
                "stage"
            ),

            "topic": metadata.get(
                "topic"
            ),

            "type": metadata.get(
                "type"
            ),

            "url": metadata.get(
                "url"
            ),

            # --------------------------------------
            # M3 SOURCE POLICY METADATA
            # --------------------------------------

            "authority_type": metadata.get(
                "authority_type",
                "OFFICIAL_GUIDANCE",
            ),

            "source_status": metadata.get(
                "source_status",
                "CURRENT",
            ),

            "publication_date": metadata.get(
                "publication_date"
            ),

            "effective_date": metadata.get(
                "effective_date"
            ),

            "last_reviewed_date": metadata.get(
                "last_reviewed_date"
            ),

            "supersedes": metadata.get(
                "supersedes",
                [],
            ),

            "superseded_by": metadata.get(
                "superseded_by"
            ),

            # --------------------------------------
            # RETRIEVAL CONTENT
            # --------------------------------------

            "text": text,
        }

        records.append(
            record
        )

    return records


# ==================================================
# MAIN
# ==================================================


def main():

    raw_files = sorted(
        RAW_DIR.glob(
            "*.json"
        )
    )

    print(
        f"Raw documents: "
        f"{len(raw_files)}"
    )

    all_chunks = []

    for file_path in raw_files:

        document = json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        chunks = chunk_document(
            document
        )

        all_chunks.extend(
            chunks
        )

        metadata = document.get(
            "metadata",
            {},
        )

        print(
            f"{metadata.get('id', file_path.stem)}: "
            f"{len(chunks)} chunks | "
            f"{metadata.get('authority_type', 'UNKNOWN')}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            all_chunks,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "========================================"
    )

    print(
        "CHUNKING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Documents: "
        f"{len(raw_files)}"
    )

    print(
        f"Total chunks: "
        f"{len(all_chunks)}"
    )

    print(
        f"Saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()