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
# PROJECT IMPORTS
# ==================================================

from src.retrieval.source_conflict import (
    choose_preferred_source,
)


# ==================================================
# TEST CASES
# ==================================================

TEST_CASES = [
    {
        "id": "status_001",
        "name": "Current beats archived",
        "source_a": {
            "source_id": "current_guidance",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "publication_date": "2024-01-01",
        },
        "source_b": {
            "source_id": "archived_guidance",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "ARCHIVED",
            "publication_date": "2025-01-01",
        },
        "expected_preferred": "A",
    },
    {
        "id": "authority_001",
        "name": "Regulation beats guidance",
        "source_a": {
            "source_id": "regulation",
            "authority_type": "REGULATION",
            "source_status": "CURRENT",
            "last_reviewed_date": "2026-01-01",
        },
        "source_b": {
            "source_id": "guidance",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "last_reviewed_date": "2026-08-01",
        },
        "expected_preferred": "A",
    },
    {
        "id": "supersession_001",
        "name": "Explicit supersession wins",
        "source_a": {
            "source_id": "new_policy",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "publication_date": "2025-01-01",
            "supersedes": [
                "old_regulation_example"
            ],
        },
        "source_b": {
            "source_id": "old_regulation_example",
            "authority_type": "REGULATION",
            "source_status": "CURRENT",
            "publication_date": "2024-01-01",
        },
        "expected_preferred": "A",
    },
    {
        "id": "date_001",
        "name": "Newer equal-authority source wins",
        "source_a": {
            "source_id": "older_guidance",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "publication_date": "2022-01-01",
        },
        "source_b": {
            "source_id": "newer_guidance",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "publication_date": "2025-01-01",
        },
        "expected_preferred": "B",
    },
    {
        "id": "tie_001",
        "name": "Equal sources produce no winner",
        "source_a": {
            "source_id": "source_a",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "publication_date": "2025-01-01",
        },
        "source_b": {
            "source_id": "source_b",
            "authority_type": "OFFICIAL_GUIDANCE",
            "source_status": "CURRENT",
            "publication_date": "2025-01-01",
        },
        "expected_preferred": None,
    },
]


# ==================================================
# RUN SINGLE TEST
# ==================================================


def run_test(
    test_case,
):
    result = choose_preferred_source(
        test_case["source_a"],
        test_case["source_b"],
    )

    actual = result[
        "preferred"
    ]

    expected = test_case[
        "expected_preferred"
    ]

    passed = (
        actual == expected
    )

    return {
        "id": test_case["id"],
        "name": test_case["name"],
        "expected": expected,
        "actual": actual,
        "reason": result[
            "reason"
        ],
        "passed": passed,
    }


# ==================================================
# MAIN
# ==================================================


def main():

    results = []

    for test_case in TEST_CASES:

        result = run_test(
            test_case
        )

        results.append(
            result
        )

        print()

        print(
            "========================================"
        )

        print(
            result["name"]
        )

        print(
            "========================================"
        )

        print(
            "Expected:",
            result["expected"],
        )

        print(
            "Actual:",
            result["actual"],
        )

        print(
            "Reason:",
            result["reason"],
        )

        print(
            "PASS"
            if result["passed"]
            else "FAIL"
        )

    passed_count = sum(
        1
        for result in results
        if result[
            "passed"
        ]
    )

    total = len(
        results
    )

    print()

    print(
        "========================================"
    )

    print(
        "M3 SOURCE POLICY EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Passed: "
        f"{passed_count}/{total}"
    )

    print(
        f"Accuracy: "
        f"{passed_count / total:.1%}"
    )

    print(
        "========================================"
    )

    report_path = (
        PROJECT_ROOT
        / "data/evaluation/source_policy_report.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "total": total,
                "passed": passed_count,
                "accuracy": (
                    passed_count / total
                ),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Saved report:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()