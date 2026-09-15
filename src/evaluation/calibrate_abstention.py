from pathlib import Path
import json


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/retrieval_report.json"
)


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_threshold(
    results,
    threshold,
):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:

        score = result.get(
            "top_retrieval_score"
        )

        expected = result[
            "expected_answerable"
        ]

        if score is None:
            predicted = False
        else:
            predicted = (
                score >= threshold
            )

        if predicted and expected:
            true_positive += 1

        elif predicted and not expected:
            false_positive += 1

        elif (
            not predicted
            and expected
        ):
            false_negative += 1

        else:
            true_negative += 1

    total = len(results)

    accuracy = safe_divide(
        true_positive
        + true_negative,
        total,
    )

    precision = safe_divide(
        true_positive,
        true_positive
        + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive
        + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative
        + false_positive,
    )

    f1 = safe_divide(
        2
        * precision
        * recall,
        precision + recall,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def print_score_distribution(
    results,
):
    answerable_scores = []
    unanswerable_scores = []

    for result in results:

        score = result.get(
            "top_retrieval_score"
        )

        if score is None:
            continue

        if result[
            "expected_answerable"
        ]:
            answerable_scores.append(
                score
            )

        else:
            unanswerable_scores.append(
                score
            )

    answerable_scores.sort()

    unanswerable_scores.sort()

    print(
        "\nANSWERABLE SCORES"
    )

    print(
        "----------------------------------------"
    )

    for score in answerable_scores:
        print(
            f"{score:.4f}"
        )

    print(
        "\nUNANSWERABLE SCORES"
    )

    print(
        "----------------------------------------"
    )

    for score in unanswerable_scores:
        print(
            f"{score:.4f}"
        )


def main():
    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    results = report[
        "results"
    ]

    print(
        "========================================"
    )

    print(
        "KK-GPT ABSTENTION CALIBRATION"
    )

    print(
        "========================================"
    )

    print_score_distribution(
        results
    )

    print(
        "\nTHRESHOLD SWEEP"
    )

    print(
        "------------------------------------------------"
    )

    print(
        "Thresh   Acc     Prec    Recall  Spec    FP   FN"
    )

    print(
        "------------------------------------------------"
    )

    evaluations = []

    threshold = 0.20

    while threshold <= 0.80:

        evaluation = evaluate_threshold(
            results,
            threshold,
        )

        evaluations.append(
            evaluation
        )

        print(
            f"{threshold:.3f}   "
            f"{evaluation['accuracy']:.1%}   "
            f"{evaluation['precision']:.1%}   "
            f"{evaluation['recall']:.1%}   "
            f"{evaluation['specificity']:.1%}   "
            f"{evaluation['false_positive']:2d}   "
            f"{evaluation['false_negative']:2d}"
        )

        threshold += 0.025

    # ----------------------------------------------
    # RECOMMENDATION
    # ----------------------------------------------
    #
    # For immigration RAG, false-positive answers
    # are more costly than unnecessary abstention.
    #
    # First require reasonably high specificity.
    # Then maximize recall among those options.
    # ----------------------------------------------

    safe_candidates = [
        item
        for item in evaluations
        if item[
            "specificity"
        ] >= 0.80
    ]

    if safe_candidates:

        recommended = max(
            safe_candidates,
            key=lambda item: (
                item["recall"],
                item["precision"],
                item["accuracy"],
            ),
        )

    else:

        recommended = max(
            evaluations,
            key=lambda item: (
                item["specificity"],
                item["recall"],
                item["accuracy"],
            ),
        )

    print()

    print(
        "========================================"
    )

    print(
        "RECOMMENDED PROVISIONAL THRESHOLD"
    )

    print(
        "========================================"
    )

    print(
        f"Threshold: "
        f"{recommended['threshold']:.3f}"
    )

    print(
        f"Accuracy: "
        f"{recommended['accuracy']:.1%}"
    )

    print(
        f"Precision: "
        f"{recommended['precision']:.1%}"
    )

    print(
        f"Recall: "
        f"{recommended['recall']:.1%}"
    )

    print(
        f"Specificity: "
        f"{recommended['specificity']:.1%}"
    )

    print(
        f"False positives: "
        f"{recommended['false_positive']}"
    )

    print(
        f"False negatives: "
        f"{recommended['false_negative']}"
    )

    print()

    print(
        "This threshold is experimental."
    )

    print(
        "Do not copy it into app.py yet."
    )


if __name__ == "__main__":
    main()