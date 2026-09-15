from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

from src.retrieval.search_with_reranker import (
    EMBEDDING_MODEL_NAME,
    understand_query,
    retrieve_candidates,
    load_local_corpus,
)

from src.generation.generate_answer import (
    ABSTAIN_MESSAGE,
    assess_evidence,
    build_evidence,
    generate_answer_text,
    validate_citations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions_gold_v1.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/m5_citation_metrics.json"
)

RETRIEVAL_TOP_K = 5


def load_questions():

    data = json.loads(
        GOLD_PATH.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(data, dict):
        data = data.get(
            "questions",
            []
        )

    return [
        item
        for item in data
        if item.get(
            "answerable",
            True,
        )
    ]


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate():

    questions = load_questions()

    chunks, embeddings = (
        load_local_corpus()
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    gate_supported = 0

    generated_answers = 0

    citation_present = 0

    citation_valid = 0

    model_abstentions = 0

    failures = []

    results_report = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item[
            "question"
        ]

        question_id = item[
            "id"
        ]

        query_info = (
            understand_query(
                question
            )
        )

        results = (
            retrieve_candidates(
                retrieval_query=(
                    query_info[
                        "retrieval_query"
                    ]
                ),
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=(
                    embedding_model
                ),
                topic_filter=(
                    query_info[
                        "topic_filter"
                    ]
                ),
                top_k=RETRIEVAL_TOP_K,
            )
        )

        evidence_check = (
            assess_evidence(
                question,
                results,
            )
        )

        row = {
            "id": question_id,
            "question": question,
            "evidence_supported": (
                evidence_check[
                    "supported"
                ]
            ),
        }

        if not evidence_check[
            "supported"
        ]:

            row[
                "status"
            ] = "EVIDENCE_ABSTENTION"

            results_report.append(
                row
            )

            print(
                f"[{index}/{len(questions)}] "
                f"{question_id}: "
                "EVIDENCE_ABSTENTION"
            )

            continue

        gate_supported += 1

        evidence_items = (
            build_evidence(
                results
            )
        )

        answer = (
            generate_answer_text(
                question,
                evidence_items,
            )
        )

        if (
            ABSTAIN_MESSAGE.lower()
            in answer.lower()
        ):

            model_abstentions += 1

            row.update(
                {
                    "status": (
                        "GENERATOR_ABSTENTION"
                    ),
                    "answer": answer,
                }
            )

            failures.append(
                row
            )

            results_report.append(
                row
            )

            print(
                f"[{index}/{len(questions)}] "
                f"{question_id}: "
                "GENERATOR_ABSTENTION"
            )

            continue

        generated_answers += 1

        citation_check = (
            validate_citations(
                answer,
                evidence_items,
            )
        )

        cited_ids = (
            citation_check.get(
                "cited_ids",
                [],
            )
        )

        has_citation = bool(
            cited_ids
        )

        if has_citation:
            citation_present += 1

        if citation_check[
            "valid"
        ]:
            citation_valid += 1

        status = (
            "PASS"
            if citation_check[
                "valid"
            ]
            else "FAIL"
        )

        row.update(
            {
                "status": status,

                "answer": answer,

                "citation_present": (
                    has_citation
                ),

                "citation_valid": (
                    citation_check[
                        "valid"
                    ]
                ),

                "cited_ids": (
                    cited_ids
                ),

                "invalid_ids": (
                    citation_check.get(
                        "invalid_ids",
                        [],
                    )
                ),

                "citation_reason": (
                    citation_check.get(
                        "reason",
                        "",
                    )
                ),

                "evidence_source_ids": [
                    evidence.get(
                        "source_id"
                    )
                    for evidence
                    in evidence_items
                ],
            }
        )

        if not citation_check[
            "valid"
        ]:

            failures.append(
                row
            )

        results_report.append(
            row
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}: {status}"
        )

    citation_presence_rate = (
        safe_divide(
            citation_present,
            generated_answers,
        )
    )

    citation_validity_rate = (
        safe_divide(
            citation_valid,
            generated_answers,
        )
    )

    report = {
        "dataset": {
            "answerable_questions": (
                len(questions)
            ),
            "evidence_supported": (
                gate_supported
            ),
            "generated_answers": (
                generated_answers
            ),
            "generator_abstentions": (
                model_abstentions
            ),
        },

        "metrics": {
            "citation_presence_rate": (
                citation_presence_rate
            ),

            "citation_id_validity_rate": (
                citation_validity_rate
            ),

            "citation_failures": (
                len(failures)
            ),
        },

        "failures": failures,

        "results": (
            results_report
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "M5 CITATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Answerable questions:      "
        f"{len(questions)}"
    )

    print(
        f"Evidence supported:        "
        f"{gate_supported}"
    )

    print(
        f"Generated answers:         "
        f"{generated_answers}"
    )

    print(
        f"Generator abstentions:     "
        f"{model_abstentions}"
    )

    print()

    print(
        f"Citation presence:         "
        f"{citation_presence_rate * 100:.1f}%"
    )

    print(
        f"Citation ID validity:      "
        f"{citation_validity_rate * 100:.1f}%"
    )

    print(
        f"Citation failures:         "
        f"{len(failures)}"
    )

    if failures:

        print(
            "\nFAILURES"
        )

        for item in failures:

            print(
                f"- {item['id']}: "
                f"{item['status']}"
            )

    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )


if __name__ == "__main__":
    evaluate()