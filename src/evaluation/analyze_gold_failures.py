from pathlib import Path
import json


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/rag_v1_gold_metrics.json"
)


# ==================================================
# LOAD
# ==================================================


def load_report():

    return json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )


# ==================================================
# DISPLAY HELPERS
# ==================================================


def print_result(
    result,
    rank,
):

    marker = (
        "GOLD"
        if result.get(
            "is_gold_relevant"
        )
        else "NOT GOLD"
    )

    print(
        f"\n  Rank {rank} [{marker}]"
    )

    print(
        f"    Source ID: "
        f"{result.get('source_id')}"
    )

    print(
        f"    Document: "
        f"{result.get('document')}"
    )

    print(
        f"    Topic: "
        f"{result.get('topic')}"
    )

    print(
        f"    Agency: "
        f"{result.get('agency')}"
    )

    print(
        f"    Authority: "
        f"{result.get('authority_type')}"
    )

    retrieval_score = result.get(
        "retrieval_score"
    )

    reranker_score = result.get(
        "reranker_score"
    )

    final_score = result.get(
        "final_score"
    )

    if retrieval_score is not None:

        print(
            "    Retrieval score: "
            f"{retrieval_score:.4f}"
        )

    if reranker_score is not None:

        print(
            "    Reranker score: "
            f"{reranker_score:.4f}"
        )

    if final_score is not None:

        print(
            "    Final score: "
            f"{final_score:.4f}"
        )


# ==================================================
# ANALYSIS
# ==================================================


def analyze():

    report = load_report()

    details = report.get(
        "details",
        []
    )

    failures = []

    for item in details:

        after = item.get(
            "after_reranking",
            {}
        )

        if (
            after.get(
                "recall_at_5"
            )
            == 0
        ):

            failures.append(
                item
            )

    print(
        "=" * 70
    )

    print(
        "RAG V1 GOLD-SOURCE FAILURE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRecall@5 failures: "
        f"{len(failures)}"
    )

    if not failures:

        print(
            "\nNo Recall@5 failures found."
        )

        return

    for number, item in enumerate(
        failures,
        start=1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FAILURE {number}"
        )

        print(
            "=" * 70
        )

        print(
            f"\nID: "
            f"{item.get('id')}"
        )

        print(
            f"Question: "
            f"{item.get('question')}"
        )

        print(
            f"Stage: "
            f"{item.get('stage')}"
        )

        print(
            f"Topic: "
            f"{item.get('topic')}"
        )

        print(
            "\nGold source IDs:"
        )

        for source_id in item.get(
            "gold_source_ids",
            []
        ):

            print(
                f"  - {source_id}"
            )

        print(
            "\nQuery understanding:"
        )

        print(
            "  Retrieval query: "
            f"{item.get('retrieval_query')}"
        )

        print(
            "  Topic filter: "
            f"{item.get('topic_filter')}"
        )

        print(
            "\nBEFORE RERANKING TOP 5"
        )

        before = item.get(
            "before_reranking",
            {}
        )

        before_results = before.get(
            "top_5",
            []
        )

        for rank, result in enumerate(
            before_results,
            start=1,
        ):

            print_result(
                result,
                rank,
            )

        print(
            "\nAFTER RERANKING TOP 5"
        )

        after = item.get(
            "after_reranking",
            {}
        )

        after_results = after.get(
            "top_5",
            []
        )

        for rank, result in enumerate(
            after_results,
            start=1,
        ):

            print_result(
                result,
                rank,
            )

        print(
            "\nEvaluation notes:"
        )

        print(
            "  "
            + str(
                item.get(
                    "evaluation_notes",
                    ""
                )
            )
        )


# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":

    analyze()