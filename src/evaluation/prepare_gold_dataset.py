from pathlib import Path
import json
import sys


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ==================================================
# PATHS
# ==================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions.json"
)

SOURCES_PATH = (
    PROJECT_ROOT
    / "config/sources.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions_gold_draft.json"
)


# ==================================================
# LOAD JSON
# ==================================================


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ==================================================
# SOURCE INDEX
# ==================================================


def build_source_index(
    sources,
):
    index = {}

    for source in sources:

        source_id = source.get(
            "id"
        )

        if not source_id:
            continue

        index[source_id] = {
            "id": source_id,
            "agency": source.get(
                "agency"
            ),
            "title": source.get(
                "title"
            ),
            "stage": source.get(
                "stage"
            ),
            "topic": source.get(
                "topic"
            ),
            "authority_type": source.get(
                "authority_type"
            ),
            "source_status": source.get(
                "source_status"
            ),
            "url": source.get(
                "url"
            ),
        }

    return index


# ==================================================
# TOPIC MATCHING
# ==================================================


def get_candidate_sources(
    question,
    source_index,
):
    """
    Candidate sources are suggestions only.

    They are NOT automatically considered gold.
    """

    expected_topic = (
        question.get(
            "expected_topic"
        )
        or question.get(
            "topic"
        )
        or ""
    ).lower()

    candidates = []

    for source in source_index.values():

        source_topic = str(
            source.get(
                "topic",
                ""
            )
        ).lower()

        source_stage = str(
            source.get(
                "stage",
                ""
            )
        ).lower()

        if (
            expected_topic
            and (
                expected_topic
                in source_topic
                or expected_topic
                in source_stage
            )
        ):
            candidates.append(
                source
            )

    return candidates


# ==================================================
# BUILD DRAFT
# ==================================================


def build_gold_draft():

    questions = load_json(
        QUESTIONS_PATH
    )

    sources = load_json(
        SOURCES_PATH
    )

    source_index = build_source_index(
        sources
    )

    draft = []

    for question in questions:

        item = dict(
            question
        )

        # ------------------------------------------
        # Only answerable questions need retrieval
        # source-level gold labels.
        # ------------------------------------------

        if item.get(
            "answerable",
            True,
        ):

            item[
                "expected_source_ids"
            ] = []

            candidates = (
                get_candidate_sources(
                    item,
                    source_index,
                )
            )

            item[
                "_candidate_sources"
            ] = [
                {
                    "source_id": source[
                        "id"
                    ],
                    "agency": source.get(
                        "agency"
                    ),
                    "title": source.get(
                        "title"
                    ),
                    "authority_type": (
                        source.get(
                            "authority_type"
                        )
                    ),
                }
                for source
                in candidates
            ]

            item[
                "_evaluation_notes"
            ] = ""

        draft.append(
            item
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            draft,
            indent=2,
        ),
        encoding="utf-8",
    )

    answerable_count = sum(
        1
        for item in draft
        if item.get(
            "answerable",
            True,
        )
    )

    print(
        "=" * 50
    )

    print(
        "GOLD DATASET DRAFT CREATED"
    )

    print(
        "=" * 50
    )

    print(
        f"Total questions: "
        f"{len(draft)}"
    )

    print(
        f"Answerable questions: "
        f"{answerable_count}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Candidate sources are suggestions only."
    )

    print(
        "Manually fill expected_source_ids."
    )

    print(
        "Do not automatically copy every "
        "candidate into the gold labels."
    )


# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":
    build_gold_draft()