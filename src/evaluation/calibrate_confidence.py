from pathlib import Path
import json
from statistics import mean, median


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/retrieval_report.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/confidence_report.json"
)


# ==================================================
# HELPERS
# ==================================================


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_signals(result):
    """
    Extract confidence-related signals from one
    evaluation result.

    Signals:
    - top retrieval similarity
    - top reranker score
    - reranker top1/top2 margin
    - number of positive reranker scores in top 3
    """

    retrieval_score = result.get(
        "top_retrieval_score"
    )

    reranked_results = result.get(
        "results",
        [],
    )

    # ----------------------------------------------
    # TOP RERANKER SCORE
    # ----------------------------------------------

    if reranked_results:
        top_reranker_score = (
            reranked_results[0].get(
                "reranker_score"
            )
        )
    else:
        top_reranker_score = None

    # ----------------------------------------------
    # RERANKER MARGIN
    #
    # Example:
    #
    # top1 = 6.0
    # top2 = 2.0
    #
    # margin = 4.0
    #
    # A larger margin can mean result #1 is
    # substantially more relevant than alternatives.
    # ----------------------------------------------

    if len(reranked_results) >= 2:

        top1 = reranked_results[
            0
        ].get(
            "reranker_score",
            0.0,
        )

        top2 = reranked_results[
            1
        ].get(
            "reranker_score",
            0.0,
        )

        reranker_margin = (
            top1 - top2
        )

    else:
        reranker_margin = None

    # ----------------------------------------------
    # POSITIVE RERANKER COUNT
    #
    # Count how many of the best 3 chunks have
    # reranker score > 0.
    #
    # More supporting chunks may indicate stronger
    # evidence.
    # ----------------------------------------------

    positive_top3 = 0

    for item in reranked_results[:3]:

        score = item.get(
            "reranker_score"
        )

        if (
            score is not None
            and score > 0
        ):
            positive_top3 += 1

    return {
        "retrieval_score": retrieval_score,
        "top_reranker_score": (
            top_reranker_score
        ),
        "reranker_margin": (
            reranker_margin
        ),
        "positive_top3": (
            positive_top3
        ),
    }


# ==================================================
# DISTRIBUTION HELPERS
# ==================================================


def describe(values):
    """
    Return simple descriptive statistics.
    """

    clean_values = [
        value
        for value in values
        if value is not None
    ]

    if not clean_values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(
            clean_values
        ),
        "min": min(
            clean_values
        ),
        "max": max(
            clean_values
        ),
        "mean": mean(
            clean_values
        ),
        "median": median(
            clean_values
        ),
    }


def print_distribution(
    title,
    answerable_values,
    unanswerable_values,
):
    answerable_stats = describe(
        answerable_values
    )

    unanswerable_stats = describe(
        unanswerable_values
    )

    print()
    print(
        "========================================"
    )

    print(
        title
    )

    print(
        "========================================"
    )

    print(
        "ANSWERABLE"
    )

    print(
        f"Count:  "
        f"{answerable_stats['count']}"
    )

    if answerable_stats["count"]:

        print(
            f"Min:    "
            f"{answerable_stats['min']:.4f}"
        )

        print(
            f"Median: "
            f"{answerable_stats['median']:.4f}"
        )

        print(
            f"Mean:   "
            f"{answerable_stats['mean']:.4f}"
        )

        print(
            f"Max:    "
            f"{answerable_stats['max']:.4f}"
        )

    print()

    print(
        "UNANSWERABLE"
    )

    print(
        f"Count:  "
        f"{unanswerable_stats['count']}"
    )

    if unanswerable_stats["count"]:

        print(
            f"Min:    "
            f"{unanswerable_stats['min']:.4f}"
        )

        print(
            f"Median: "
            f"{unanswerable_stats['median']:.4f}"
        )

        print(
            f"Mean:   "
            f"{unanswerable_stats['mean']:.4f}"
        )

        print(
            f"Max:    "
            f"{unanswerable_stats['max']:.4f}"
        )


# ==================================================
# GATE EVALUATION
# ==================================================


def evaluate_gate(
    rows,
    retrieval_threshold,
    reranker_threshold,
):
    """
    Combined confidence rule:

    ANSWER only if:

        retrieval >= retrieval_threshold

    AND

        top reranker >= reranker_threshold

    This is deliberately simple.

    We first want to learn whether combining the
    signals is better than raw retrieval alone.
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for row in rows:

        retrieval_score = row[
            "signals"
        ][
            "retrieval_score"
        ]

        reranker_score = row[
            "signals"
        ][
            "top_reranker_score"
        ]

        expected = row[
            "expected_answerable"
        ]

        if (
            retrieval_score is None
            or reranker_score is None
        ):
            predicted = False

        else:
            predicted = (
                retrieval_score
                >= retrieval_threshold
                and reranker_score
                >= reranker_threshold
            )

        if predicted and expected:
            tp += 1

        elif predicted and not expected:
            fp += 1

        elif (
            not predicted
            and expected
        ):
            fn += 1

        else:
            tn += 1

    total = len(rows)

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    f1 = safe_divide(
        2
        * precision
        * recall,
        precision + recall,
    )

    return {
        "retrieval_threshold": (
            retrieval_threshold
        ),
        "reranker_threshold": (
            reranker_threshold
        ),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# ==================================================
# GRID SEARCH
# ==================================================


def run_grid_search(
    rows,
):
    """
    Try combinations of retrieval and reranker
    thresholds.

    We intentionally keep this small and readable.
    """

    retrieval_thresholds = [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    reranker_thresholds = [
        -4.0,
        -2.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    evaluations = []

    for retrieval_threshold in (
        retrieval_thresholds
    ):

        for reranker_threshold in (
            reranker_thresholds
        ):

            evaluation = evaluate_gate(
                rows,
                retrieval_threshold,
                reranker_threshold,
            )

            evaluations.append(
                evaluation
            )

    return evaluations


# ==================================================
# SELECT CANDIDATES
# ==================================================


def select_candidates(
    evaluations,
):
    """
    For immigration RAG we care strongly about
    false positives.

    Show candidate rules with:

        specificity >= 80%

    Then rank them by:
        1. recall
        2. precision
        3. accuracy
    """

    safe = [
        item
        for item in evaluations
        if item[
            "specificity"
        ] >= 0.80
    ]

    safe.sort(
        key=lambda item: (
            item["recall"],
            item["precision"],
            item["accuracy"],
        ),
        reverse=True,
    )

    return safe[:10]


# ==================================================
# PRINT ROWS
# ==================================================


def print_question_signals(
    rows,
):
    print()

    print(
        "========================================"
    )

    print(
        "QUESTION SIGNALS"
    )

    print(
        "========================================"
    )

    for row in rows:

        signals = row[
            "signals"
        ]

        label = (
            "ANSWERABLE"
            if row[
                "expected_answerable"
            ]
            else "UNANSWERABLE"
        )

        print()

        print(
            f"[{label}] "
            f"{row['id']}"
        )

        print(
            row[
                "question"
            ]
        )

        retrieval_score = signals[
            "retrieval_score"
        ]

        reranker_score = signals[
            "top_reranker_score"
        ]

        margin = signals[
            "reranker_margin"
        ]

        positive_top3 = signals[
            "positive_top3"
        ]

        if retrieval_score is not None:

            print(
                f"Retrieval: "
                f"{retrieval_score:.4f}"
            )

        if reranker_score is not None:

            print(
                f"Reranker: "
                f"{reranker_score:.4f}"
            )

        if margin is not None:

            print(
                f"Reranker margin: "
                f"{margin:.4f}"
            )

        print(
            f"Positive top-3 chunks: "
            f"{positive_top3}"
        )


# ==================================================
# PRINT GRID RESULTS
# ==================================================


def print_candidates(
    candidates,
):
    print()

    print(
        "========================================"
    )

    print(
        "BEST COMBINED GATES"
    )

    print(
        "========================================"
    )

    if not candidates:

        print(
            "No rule reached 80% specificity."
        )

        return

    print(
        "Ret   Rerank   Acc    Prec   "
        "Recall  Spec    FP  FN"
    )

    print(
        "----------------------------------------"
    )

    for item in candidates:

        print(
            f"{item['retrieval_threshold']:.2f}  "
            f"{item['reranker_threshold']:6.2f}  "
            f"{item['accuracy']:.1%}  "
            f"{item['precision']:.1%}  "
            f"{item['recall']:.1%}  "
            f"{item['specificity']:.1%}  "
            f"{item['false_positive']:2d}  "
            f"{item['false_negative']:2d}"
        )


# ==================================================
# MAIN
# ==================================================


def main():
    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    results = report[
        "results"
    ]

    rows = []

    for result in results:

        signals = get_signals(
            result
        )

        rows.append(
            {
                "id": result[
                    "id"
                ],
                "question": result[
                    "question"
                ],
                "expected_answerable": result[
                    "expected_answerable"
                ],
                "signals": signals,
            }
        )

    # ----------------------------------------------
    # QUESTION-BY-QUESTION SIGNALS
    # ----------------------------------------------

    print_question_signals(
        rows
    )

    # ----------------------------------------------
    # SPLIT ANSWERABLE / UNANSWERABLE
    # ----------------------------------------------

    answerable = [
        row
        for row in rows
        if row[
            "expected_answerable"
        ]
    ]

    unanswerable = [
        row
        for row in rows
        if not row[
            "expected_answerable"
        ]
    ]

    # ----------------------------------------------
    # RETRIEVAL SCORE DISTRIBUTION
    # ----------------------------------------------

    print_distribution(
        "TOP RETRIEVAL SCORE",
        [
            row["signals"][
                "retrieval_score"
            ]
            for row in answerable
        ],
        [
            row["signals"][
                "retrieval_score"
            ]
            for row in unanswerable
        ],
    )

    # ----------------------------------------------
    # RERANKER SCORE DISTRIBUTION
    # ----------------------------------------------

    print_distribution(
        "TOP RERANKER SCORE",
        [
            row["signals"][
                "top_reranker_score"
            ]
            for row in answerable
        ],
        [
            row["signals"][
                "top_reranker_score"
            ]
            for row in unanswerable
        ],
    )

    # ----------------------------------------------
    # RERANKER MARGIN DISTRIBUTION
    # ----------------------------------------------

    print_distribution(
        "RERANKER TOP1-TOP2 MARGIN",
        [
            row["signals"][
                "reranker_margin"
            ]
            for row in answerable
        ],
        [
            row["signals"][
                "reranker_margin"
            ]
            for row in unanswerable
        ],
    )

    # ----------------------------------------------
    # COMBINED GATE SEARCH
    # ----------------------------------------------

    evaluations = run_grid_search(
        rows
    )

    candidates = select_candidates(
        evaluations
    )

    print_candidates(
        candidates
    )

    # ----------------------------------------------
    # SAVE REPORT
    # ----------------------------------------------

    output = {
        "rows": rows,
        "combined_gate_results": (
            evaluations
        ),
        "recommended_candidates": (
            candidates
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Detailed confidence report saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Do not change app.py yet."
    )


if __name__ == "__main__":
    main()