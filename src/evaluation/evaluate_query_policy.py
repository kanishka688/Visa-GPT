from pathlib import Path
import json
import sys
from collections import Counter


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
# PROJECT IMPORT
# ==================================================

from src.retrieval.query_policy import (
    classify_query,
)


# ==================================================
# PATHS
# ==================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/query_policy_questions.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/query_policy_report.json"
)


# ==================================================
# SINGLE QUESTION EVALUATION
# ==================================================


def evaluate_question(
    test_case,
):
    question = test_case[
        "question"
    ]

    expected_category = test_case[
        "expected_category"
    ]

    expected_route = test_case[
        "route_to_rag"
    ]

    classifier_result = classify_query(
        question
    )

    predicted_category = (
        classifier_result[
            "category"
        ]
    )

    predicted_route = (
        classifier_result[
            "route_to_rag"
        ]
    )

    category_correct = (
        predicted_category
        == expected_category
    )

    route_correct = (
        predicted_route
        == expected_route
    )

    return {
        "id": test_case["id"],
        "question": question,
        "expected_category": (
            expected_category
        ),
        "predicted_category": (
            predicted_category
        ),
        "category_correct": (
            category_correct
        ),
        "expected_route_to_rag": (
            expected_route
        ),
        "predicted_route_to_rag": (
            predicted_route
        ),
        "route_correct": (
            route_correct
        ),
        "reason": classifier_result.get(
            "reason",
            "",
        ),
    }


# ==================================================
# SUMMARY
# ==================================================


def calculate_summary(
    results,
):
    total = len(
        results
    )

    category_correct = sum(
        1
        for result in results
        if result[
            "category_correct"
        ]
    )

    route_correct = sum(
        1
        for result in results
        if result[
            "route_correct"
        ]
    )

    true_positive = sum(
        1
        for result in results
        if (
            result[
                "expected_route_to_rag"
            ]
            and result[
                "predicted_route_to_rag"
            ]
        )
    )

    true_negative = sum(
        1
        for result in results
        if (
            not result[
                "expected_route_to_rag"
            ]
            and not result[
                "predicted_route_to_rag"
            ]
        )
    )

    false_positive = sum(
        1
        for result in results
        if (
            not result[
                "expected_route_to_rag"
            ]
            and result[
                "predicted_route_to_rag"
            ]
        )
    )

    false_negative = sum(
        1
        for result in results
        if (
            result[
                "expected_route_to_rag"
            ]
            and not result[
                "predicted_route_to_rag"
            ]
        )
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        )
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        )
        else 0.0
    )

    specificity = (
        true_negative
        / (
            true_negative
            + false_positive
        )
        if (
            true_negative
            + false_positive
        )
        else 0.0
    )

    expected_category_counts = Counter(
        result[
            "expected_category"
        ]
        for result in results
    )

    predicted_category_counts = Counter(
        result[
            "predicted_category"
        ]
        for result in results
    )

    per_category = {}

    for category in sorted(
        expected_category_counts.keys()
    ):

        category_results = [
            result
            for result in results
            if result[
                "expected_category"
            ] == category
        ]

        correct = sum(
            1
            for result in category_results
            if result[
                "category_correct"
            ]
        )

        per_category[
            category
        ] = {
            "total": len(
                category_results
            ),
            "correct": correct,
            "accuracy": (
                correct
                / len(category_results)
                if category_results
                else 0.0
            ),
        }

    return {
        "total_questions": total,
        "category_correct": (
            category_correct
        ),
        "category_accuracy": (
            category_correct / total
            if total
            else 0.0
        ),
        "route_correct": (
            route_correct
        ),
        "route_accuracy": (
            route_correct / total
            if total
            else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "true_positive": (
            true_positive
        ),
        "true_negative": (
            true_negative
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
        "expected_category_counts": dict(
            expected_category_counts
        ),
        "predicted_category_counts": dict(
            predicted_category_counts
        ),
        "per_category": (
            per_category
        ),
    }


# ==================================================
# PRINTING
# ==================================================


def print_result(
    result,
):
    route_status = (
        "PASS"
        if result[
            "route_correct"
        ]
        else "FAIL"
    )

    category_status = (
        "PASS"
        if result[
            "category_correct"
        ]
        else "FAIL"
    )

    print()

    print(
        f"[ROUTE {route_status}] "
        f"[CATEGORY {category_status}] "
        f"{result['id']}"
    )

    print(
        f"Question: "
        f"{result['question']}"
    )

    print(
        f"Expected category: "
        f"{result['expected_category']}"
    )

    print(
        f"Predicted category: "
        f"{result['predicted_category']}"
    )

    print(
        f"Expected RAG route: "
        f"{result['expected_route_to_rag']}"
    )

    print(
        f"Predicted RAG route: "
        f"{result['predicted_route_to_rag']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )


def print_summary(
    summary,
):
    print()

    print(
        "========================================"
    )

    print(
        "KK-GPT HARD QUERY POLICY EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Total questions: "
        f"{summary['total_questions']}"
    )

    print()

    print(
        "CATEGORY CLASSIFICATION"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Correct: "
        f"{summary['category_correct']}"
        f"/{summary['total_questions']}"
    )

    print(
        f"Accuracy: "
        f"{summary['category_accuracy']:.1%}"
    )

    print()

    print(
        "ROUTING"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Correct: "
        f"{summary['route_correct']}"
        f"/{summary['total_questions']}"
    )

    print(
        f"Accuracy: "
        f"{summary['route_accuracy']:.1%}"
    )

    print(
        f"Precision: "
        f"{summary['precision']:.1%}"
    )

    print(
        f"Recall: "
        f"{summary['recall']:.1%}"
    )

    print(
        f"Specificity: "
        f"{summary['specificity']:.1%}"
    )

    print()

    print(
        f"True positives: "
        f"{summary['true_positive']}"
    )

    print(
        f"True negatives: "
        f"{summary['true_negative']}"
    )

    print(
        f"False positives: "
        f"{summary['false_positive']}"
    )

    print(
        f"False negatives: "
        f"{summary['false_negative']}"
    )

    print()

    print(
        "PER-CATEGORY ACCURACY"
    )

    print(
        "----------------------------------------"
    )

    for category, stats in (
        summary[
            "per_category"
        ].items()
    ):

        print(
            f"{category}: "
            f"{stats['correct']}"
            f"/{stats['total']} "
            f"({stats['accuracy']:.1%})"
        )

    print(
        "========================================"
    )


# ==================================================
# MAIN
# ==================================================


def main():
    questions = json.loads(
        QUESTIONS_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Loaded "
        f"{len(questions)} questions."
    )

    results = []

    for number, test_case in enumerate(
        questions,
        start=1,
    ):

        print()

        print(
            "----------------------------------------"
        )

        print(
            f"Question "
            f"{number}/{len(questions)}"
        )

        result = evaluate_question(
            test_case
        )

        results.append(
            result
        )

        print_result(
            result
        )

    summary = calculate_summary(
        results
    )

    report = {
        "summary": summary,
        "results": results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(
        summary
    )

    print()

    print(
        "Detailed report saved to:"
    )

    print(
        REPORT_PATH
    )


if __name__ == "__main__":
    main()