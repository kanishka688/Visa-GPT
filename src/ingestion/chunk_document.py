from pathlib import Path
import json


INPUT_PATH = Path(
    "data/processed/uscis_stem_opt_policy_alert.txt"
)

OUTPUT_PATH = Path(
    "data/processed/uscis_stem_opt_policy_alert_chunks.json"
)

MAX_CHUNK_SIZE = 1200


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = text.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def create_chunks(paragraphs: list[str]) -> list[str]:
    chunks = []
    current_chunk = []

    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if (
            current_chunk
            and current_length + paragraph_length > MAX_CHUNK_SIZE
        ):
            chunks.append("\n\n".join(current_chunk))

            current_chunk = []
            current_length = 0

        current_chunk.append(paragraph)
        current_length += paragraph_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def main():
    text = INPUT_PATH.read_text(encoding="utf-8")

    paragraphs = split_into_paragraphs(text)

    chunks = create_chunks(paragraphs)

    records = []

    for index, chunk in enumerate(chunks):
        records.append(
            {
                "chunk_id": f"uscis_stem_opt_{index:03d}",
                "source": "USCIS",
                "document": "STEM OPT Policy Alert",
                "topic": "STEM OPT",
                "published_date": "2024-08-27",
                "chunk_index": index,
                "url": (
                    "https://www.uscis.gov/sites/default/files/"
                    "document/policy-manual-updates/"
                    "20240827-STEMOPT.pdf"
                ),
                "text": chunk,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Paragraphs found: {len(paragraphs)}")
    print(f"Created {len(records)} chunks")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()