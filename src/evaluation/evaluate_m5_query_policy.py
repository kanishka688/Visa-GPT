from pathlib import Path
import json
from collections import defaultdict

from src.retrieval.query_policy import (
    classify_query,
    ALLOWED_CATEGORIES,
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

GOLD_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions_gold_v1.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/m5_query_policy_metrics.json"
)


# ==================================================
# LOAD DATASET
# ==================================================


def load_dataset():

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

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Gold dataset must be a list "
            "or contain a 'questions' list."
        )

    return data


# ==================================================
# GOLD LABEL HELPERS
# ==================================================


def get_expected_category(
    item,
):
    """
    Prefer an explicit expected_policy_category.

    For older answerable questions that may not
    yet contain the field, OFFICIAL_FACT can be
    inferred safely because answerable factual
    immigration questions are the only questions
    routed into RAG.

    We do NOT invent an exact category for an
    unlabeled unanswerable question.
    """

    explicit = item.get(
        "expected_policy_category"
    )

    if explicit:
        return explicit

    if item.get(
        "answerable",
        True,
    ):
        return "OFFICIAL_FACT"

    return None


def get_expected_route(
    item,
):
    """
    Route gold is determined independently from
    the classifier.

    answerable=True  -> route to RAG
    answerable=False -> do not route to RAG
    """

    return bool(
        item.get(
            "answerable",
            True,
        )
    )


# ==================================================
# SAFE DIVISION
# ==================================================


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# ==================================================
# EVALUATION
# ==================================================


def evaluate():

    questions = (
        load_dataset()
    )

    print(
        "=" * 60
    )

    print(
        "M5 QUERY POLICY EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Questions: {len(questions)}"
    )

    # --------------------------------------------------
    # COUNTERS
    # --------------------------------------------------

    route_correct = 0

    category_correct = 0
    category_labeled = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    confusion = defaultdict(
        lambda: defaultdict(int)
    )

    failures = []

    results = []

    unlabeled_categories = []

    # --------------------------------------------------
    # RUN CLASSIFIER
    # --------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item.get(
            "id",
            f"question_{index}",
        )

        question = item[
            "question"
        ]

        expected_route = (
            get_expected_route(
                item
            )
        )

        expected_category = (
            get_expected_category(
                item
            )
        )

        prediction = (
            classify_query(
                question
            )
        )

        predicted_category = (
            prediction.get(
                "category",
                "UNKNOWN",
            )
        )

        predicted_route = bool(
            prediction.get(
                "route_to_rag",
                False,
            )
        )

        # --------------------------------------------------
        # VALIDATE MODEL OUTPUT
        # --------------------------------------------------

        if (
            predicted_category
            not in ALLOWED_CATEGORIES
        ):

            predicted_category = (
                "UNKNOWN"
            )

            predicted_route = False

        # --------------------------------------------------
        # ROUTE METRIC
        # --------------------------------------------------

        route_match = (
            predicted_route
            == expected_route
        )

        if route_match:

            route_correct += 1

        # --------------------------------------------------
        # BINARY OFFICIAL_FACT METRICS
        # --------------------------------------------------

        if (
            expected_route
            and predicted_route
        ):

            true_positive += 1

        elif (
            not expected_route
            and predicted_route
        ):

            false_positive += 1

        elif (
            expected_route
            and not predicted_route
        ):

            false_negative += 1

        else:

            true_negative += 1

        # --------------------------------------------------
        # EXACT CATEGORY METRIC
        # --------------------------------------------------

        category_match = None

        if expected_category is not None:

            category_labeled += 1

            category_match = (
                predicted_category
                == expected_category
            )

            if category_match:

                category_correct += 1

            confusion[
                expected_category
            ][
                predicted_category
            ] += 1

        else:

            unlabeled_categories.append(
                {
                    "id": question_id,
                    "question": question,
                }
            )

        # --------------------------------------------------
        # FAILURE TRACKING
        # --------------------------------------------------

        if (
            not route_match
            or (
                category_match
                is False
            )
        ):

            failures.append(
                {
                    "id": question_id,
                    "question": question,

                    "expected_category": (
                        expected_category
                    ),

                    "predicted_category": (
                        predicted_category
                    ),

                    "expected_route": (
                        expected_route
                    ),

                    "predicted_route": (
                        predicted_route
                    ),

                    "route_correct": (
                        route_match
                    ),

                    "category_correct": (
                        category_match
                    ),
                }
            )

        # --------------------------------------------------
        # STORE RESULT
        # --------------------------------------------------

        results.append(
            {
                "id": question_id,
                "question": question,

                "expected_category": (
                    expected_category
                ),

                "predicted_category": (
                    predicted_category
                ),

                "expected_route": (
                    expected_route
                ),

                "predicted_route": (
                    predicted_route
                ),

                "route_correct": (
                    route_match
                ),

                "category_correct": (
                    category_match
                ),
            }
        )

        # --------------------------------------------------
        # TERMINAL OUTPUT
        # --------------------------------------------------

        route_icon = (
            "PASS"
            if route_match
            else "FAIL"
        )

        if category_match is None:

            category_icon = (
                "UNLABELED"
            )

        else:

            category_icon = (
                "PASS"
                if category_match
                else "FAIL"
            )

        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_id}"
        )

        print(
            question
        )

        print(
            "Expected category: "
            f"{expected_category}"
        )

        print(
            "Predicted category: "
            f"{predicted_category}"
        )

        print(
            "Expected route: "
            f"{expected_route}"
        )

        print(
            "Predicted route: "
            f"{predicted_route}"
        )

        print(
            f"Category: {category_icon}"
        )

        print(
            f"Route:    {route_icon}"
        )

    # ==================================================
    # AGGREGATE METRICS
    # ==================================================

    total = len(
        questions
    )

    route_accuracy = (
        safe_divide(
            route_correct,
            total,
        )
    )

    category_accuracy = (
        safe_divide(
            category_correct,
            category_labeled,
        )
    )

    precision = (
        safe_divide(
            true_positive,
            (
                true_positive
                + false_positive
            ),
        )
    )

    recall = (
        safe_divide(
            true_positive,
            (
                true_positive
                + false_negative
            ),
        )
    )

    specificity = (
        safe_divide(
            true_negative,
            (
                true_negative
                + false_positive
            ),
        )
    )

    f1 = (
        safe_divide(
            2
            * precision
            * recall,
            precision
            + recall,
        )
    )

    # ==================================================
    # REPORT
    # ==================================================

    report = {

        "dataset": {
            "total_questions": (
                total
            ),

            "category_labeled_questions": (
                category_labeled
            ),

            "category_unlabeled_questions": (
                len(
                    unlabeled_categories
                )
            ),
        },

        "metrics": {

            "category_accuracy": (
                category_accuracy
            ),

            "route_accuracy": (
                route_accuracy
            ),

            "official_fact_precision": (
                precision
            ),

            "official_fact_recall": (
                recall
            ),

            "official_fact_specificity": (
                specificity
            ),

            "official_fact_f1": (
                f1
            ),

            "true_positive": (
                true_positive
            ),

            "false_positive": (
                false_positive
            ),

            "false_negative": (
                false_negative
            ),

            "true_negative": (
                true_negative
            ),
        },

        "confusion_matrix": {
            expected: dict(
                predictions
            )
            for (
                expected,
                predictions
            )
            in confusion.items()
        },

        "failures": (
            failures
        ),

        "unlabeled_categories": (
            unlabeled_categories
        ),

        "results": (
            results
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

    # ==================================================
    # FINAL OUTPUT
    # ==================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "M5 QUERY POLICY RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Total questions:      "
        f"{total}"
    )

    print(
        f"Category labeled:     "
        f"{category_labeled}"
    )

    print(
        f"Category unlabeled:   "
        f"{len(unlabeled_categories)}"
    )

    print()

    print(
        f"Category accuracy:    "
        f"{category_accuracy * 100:.1f}%"
    )

    print(
        f"Route accuracy:       "
        f"{route_accuracy * 100:.1f}%"
    )

    print()

    print(
        "OFFICIAL_FACT / RAG ROUTING"
    )

    print(
        f"Precision:            "
        f"{precision * 100:.1f}%"
    )

    print(
        f"Recall:               "
        f"{recall * 100:.1f}%"
    )

    print(
        f"Specificity:          "
        f"{specificity * 100:.1f}%"
    )

    print(
        f"F1:                   "
        f"{f1 * 100:.1f}%"
    )

    print()

    print(
        "CONFUSION COUNTS"
    )

    print(
        f"TP: {true_positive}"
    )

    print(
        f"FP: {false_positive}"
    )

    print(
        f"FN: {false_negative}"
    )

    print(
        f"TN: {true_negative}"
    )

    print(
        f"\nFailures: "
        f"{len(failures)}"
    )

    if failures:

        print(
            "\nFAILED QUESTIONS"
        )

        for failure in failures:

            print(
                "\n"
                f"- {failure['id']}"
            )

            print(
                "  "
                f"{failure['question']}"
            )

            print(
                "  Expected category: "
                f"{failure['expected_category']}"
            )

            print(
                "  Predicted category: "
                f"{failure['predicted_category']}"
            )

            print(
                "  Expected route: "
                f"{failure['expected_route']}"
            )

            print(
                "  Predicted route: "
                f"{failure['predicted_route']}"
            )

    if unlabeled_categories:

        print(
            "\nUNLABELED EXACT CATEGORIES"
        )

        print(
            "These questions still participate "
            "in route metrics, but not exact "
            "category accuracy:"
        )

        for item in unlabeled_categories:

            print(
                f"- {item['id']}: "
                f"{item['question']}"
            )

    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )


# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":

    evaluate()