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
    assess_evidence,
    MAX_EVIDENCE_SOURCES,
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
    / "data/evaluation/m5_abstention_metrics.json"
)


# ==================================================
# CONFIG
# ==================================================

RETRIEVAL_TOP_K = 5


# ==================================================
# TRUE OUT-OF-CORPUS FACT QUESTIONS
# ==================================================

# IMPORTANT:
#
# "unsupported" means:
#
# The CURRENT RAG V1 CORPUS does not contain enough
# evidence to answer the exact question.
#
# It does NOT merely mean:
#
# "outside the intended product scope."
#
# Several immigration regulations/forms contain
# incidental references to other statuses. Those
# must not be labeled unsupported if the retrieved
# text genuinely answers the question.

UNSUPPORTED_OFFICIAL_FACTS = [

    {
        "id": "unsupported_001",
        "question": (
            "What are the basic eligibility "
            "requirements for U.S. naturalization?"
        ),
    },

    {
        "id": "unsupported_002",
        "question": (
            "What is Form N-400 used for?"
        ),
    },

    {
        "id": "unsupported_003",
        "question": (
            "What is Form N-600 used for?"
        ),
    },

    {
        "id": "unsupported_004",
        "question": (
            "What is Form N-600K used for?"
        ),
    },

    {
        "id": "unsupported_005",
        "question": (
            "What is Form I-130 used for?"
        ),
    },

    {
        "id": "unsupported_006",
        "question": (
            "What is Form I-751 used for?"
        ),
    },

    {
        "id": "unsupported_007",
        "question": (
            "What is Form I-864 used for?"
        ),
    },

    {
        "id": "unsupported_008",
        "question": (
            "What is Form DS-260 used for?"
        ),
    },

    {
        "id": "unsupported_009",
        "question": (
            "What is Form I-485 used for?"
        ),
    },

    {
        "id": "unsupported_010",
        "question": (
            "What is the naturalization civics test?"
        ),
    },

    {
        "id": "unsupported_011",
        "question": (
            "What English language requirements "
            "apply to naturalization applicants?"
        ),
    },

    {
        "id": "unsupported_012",
        "question": (
            "What does good moral character mean "
            "for naturalization eligibility?"
        ),
    },

    {
        "id": "unsupported_013",
        "question": (
            "When can a child automatically acquire "
            "U.S. citizenship through a parent?"
        ),
    },

    {
        "id": "unsupported_014",
        "question": (
            "What is a Certificate of Citizenship?"
        ),
    },

    {
        "id": "unsupported_015",
        "question": (
            "What documents are required when "
            "filing Form I-751 to remove conditions "
            "on permanent residence?"
        ),
    },
]


# ==================================================
# DATASET
# ==================================================


def load_supported_questions():

    data = json.loads(
        GOLD_PATH.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(
        data,
        dict,
    ):
        data = data.get(
            "questions",
            []
        )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Gold dataset must be a JSON list "
            "or contain a 'questions' list."
        )

    supported = []

    for item in data:

        if not item.get(
            "answerable",
            True,
        ):
            continue

        expected_sources = (
            item.get(
                "expected_source_ids",
                []
            )
        )

        if not expected_sources:

            raise ValueError(
                f"{item.get('id')}: "
                "answerable question has no "
                "expected_source_ids."
            )

        supported.append(
            {
                "id": item["id"],

                "question": (
                    item["question"]
                ),

                "expected_source_ids": (
                    expected_sources
                ),

                "expected_supported": True,
            }
        )

    return supported


# ==================================================
# HELPERS
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


def get_source_ids(
    results,
):

    return [
        result.get(
            "chunk",
            {},
        ).get(
            "source_id"
        )
        for result in results
    ]


def first_gold_source_rank(
    results,
    expected_source_ids,
):

    expected = set(
        expected_source_ids
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        source_id = (
            result.get(
                "chunk",
                {},
            ).get(
                "source_id"
            )
        )

        if source_id in expected:
            return rank

    return None


# ==================================================
# RETRIEVAL
# ==================================================


def retrieve(
    question,
    chunks,
    embeddings,
    embedding_model,
):

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

            top_k=(
                RETRIEVAL_TOP_K
            ),
        )
    )

    return (
        query_info,
        results,
    )


# ==================================================
# EVALUATION
# ==================================================


def evaluate():

    supported_questions = (
        load_supported_questions()
    )

    unsupported_questions = [

        {
            **item,

            "expected_source_ids": [],

            "expected_supported": False,
        }

        for item
        in UNSUPPORTED_OFFICIAL_FACTS
    ]

    dataset = (
        supported_questions
        + unsupported_questions
    )

    print(
        "=" * 60
    )

    print(
        "M5 EVIDENCE / ABSTENTION EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Supported questions:   "
        f"{len(supported_questions)}"
    )

    print(
        f"Unsupported questions: "
        f"{len(unsupported_questions)}"
    )

    print(
        f"Total:                 "
        f"{len(dataset)}"
    )

    print(
        f"Evidence budget:       "
        f"top {MAX_EVIDENCE_SOURCES}"
    )

    print(
        f"Retrieval depth:       "
        f"top {RETRIEVAL_TOP_K}"
    )

    # ==================================================
    # LOAD CORPUS
    # ==================================================

    print(
        "\nLoading corpus..."
    )

    chunks, embeddings = (
        load_local_corpus()
    )

    print(
        "Loading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    # ==================================================
    # COUNTERS
    # ==================================================

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    failures = []

    results_report = []

    # ==================================================
    # RUN
    # ==================================================

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        question_id = (
            item["id"]
        )

        question = (
            item["question"]
        )

        expected_supported = bool(
            item[
                "expected_supported"
            ]
        )

        expected_sources = (
            item.get(
                "expected_source_ids",
                []
            )
        )

        query_info, results = (
            retrieve(
                question=question,

                chunks=chunks,

                embeddings=embeddings,

                embedding_model=(
                    embedding_model
                ),
            )
        )

        evidence_check = (
            assess_evidence(
                question,
                results,
            )
        )

        predicted_supported = bool(
            evidence_check.get(
                "supported",
                False,
            )
        )

        correct = (
            predicted_supported
            == expected_supported
        )

        # ==================================================
        # CONFUSION MATRIX
        # ==================================================

        if (
            expected_supported
            and predicted_supported
        ):

            true_positive += 1

        elif (
            not expected_supported
            and predicted_supported
        ):

            false_positive += 1

        elif (
            expected_supported
            and not predicted_supported
        ):

            false_negative += 1

        else:

            true_negative += 1

        # ==================================================
        # DIAGNOSTICS
        # ==================================================

        retrieved_source_ids = (
            get_source_ids(
                results
            )
        )

        gold_source_rank = None

        if expected_supported:

            gold_source_rank = (
                first_gold_source_rank(
                    results,
                    expected_sources,
                )
            )

        failure_type = None

        if not correct:

            if expected_supported:

                if (
                    gold_source_rank
                    is None
                ):

                    failure_type = (
                        "NO_GOLD_SOURCE_IN_TOP5"
                    )

                elif (
                    gold_source_rank
                    > MAX_EVIDENCE_SOURCES
                ):

                    failure_type = (
                        "GOLD_SOURCE_OUTSIDE_"
                        "EVIDENCE_BUDGET"
                    )

                else:

                    failure_type = (
                        "GOLD_SOURCE_IN_BUDGET_"
                        "BUT_GATE_REJECTED"
                    )

            else:

                failure_type = (
                    "FALSE_SUPPORT_ON_"
                    "UNSUPPORTED_QUERY"
                )

        row = {

            "id": (
                question_id
            ),

            "question": (
                question
            ),

            "expected_supported": (
                expected_supported
            ),

            "predicted_supported": (
                predicted_supported
            ),

            "correct": (
                correct
            ),

            "reason": (
                evidence_check.get(
                    "reason",
                    "",
                )
            ),

            "topic_filter": (
                query_info.get(
                    "topic_filter"
                )
            ),

            "retrieval_query": (
                query_info.get(
                    "retrieval_query"
                )
            ),

            "expected_source_ids": (
                expected_sources
            ),

            "retrieved_source_ids": (
                retrieved_source_ids
            ),

            "first_gold_source_rank": (
                gold_source_rank
            ),

            "evidence_source_ids": (
                retrieved_source_ids[
                    :MAX_EVIDENCE_SOURCES
                ]
            ),

            "failure_type": (
                failure_type
            ),
        }

        results_report.append(
            row
        )

        if not correct:

            failures.append(
                row
            )

        status = (
            "PASS"
            if correct
            else "FAIL"
        )

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{question_id} — {status}"
        )

        print(
            question
        )

        print(
            "Expected supported: "
            f"{expected_supported}"
        )

        print(
            "Predicted supported: "
            f"{predicted_supported}"
        )

        print(
            "Reason: "
            f"{evidence_check.get('reason', '')}"
        )

        print(
            "Topic filter: "
            f"{query_info.get('topic_filter')}"
        )

        if expected_supported:

            print(
                "First gold source rank: "
                f"{gold_source_rank}"
            )

        print(
            "Evidence sources:"
        )

        for source_id in (
            retrieved_source_ids[
                :MAX_EVIDENCE_SOURCES
            ]
        ):

            print(
                f"  - {source_id}"
            )

        if failure_type:

            print(
                "Failure type: "
                f"{failure_type}"
            )

    # ==================================================
    # METRICS
    # ==================================================

    total = len(
        dataset
    )

    supported_total = len(
        supported_questions
    )

    unsupported_total = len(
        unsupported_questions
    )

    abstention_accuracy = (
        safe_divide(
            (
                true_positive
                + true_negative
            ),
            total,
        )
    )

    supported_recall = (
        safe_divide(
            true_positive,
            (
                true_positive
                + false_negative
            ),
        )
    )

    unsupported_specificity = (
        safe_divide(
            true_negative,
            (
                true_negative
                + false_positive
            ),
        )
    )

    precision_when_supported = (
        safe_divide(
            true_positive,
            (
                true_positive
                + false_positive
            ),
        )
    )

    false_answer_rate = (
        safe_divide(
            false_positive,
            unsupported_total,
        )
    )

    false_abstention_rate = (
        safe_divide(
            false_negative,
            supported_total,
        )
    )

    # ==================================================
    # REPORT
    # ==================================================

    report = {

        "dataset": {

            "supported_questions": (
                supported_total
            ),

            "unsupported_questions": (
                unsupported_total
            ),

            "total_questions": (
                total
            ),
        },

        "config": {

            "retrieval_top_k": (
                RETRIEVAL_TOP_K
            ),

            "evidence_top_k": (
                MAX_EVIDENCE_SOURCES
            ),

            "reranker_enabled": False,

            "source_diversity_enabled": True,

            "max_chunks_per_source": 2,
        },

        "metrics": {

            "abstention_accuracy": (
                abstention_accuracy
            ),

            "supported_question_recall": (
                supported_recall
            ),

            "unsupported_question_specificity": (
                unsupported_specificity
            ),

            "precision_when_supported": (
                precision_when_supported
            ),

            "false_answer_rate": (
                false_answer_rate
            ),

            "false_abstention_rate": (
                false_abstention_rate
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

        "failures": (
            failures
        ),

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

    # ==================================================
    # FINAL RESULTS
    # ==================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "M5 ABSTENTION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Supported questions:   "
        f"{supported_total}"
    )

    print(
        f"Unsupported questions: "
        f"{unsupported_total}"
    )

    print(
        f"Total questions:       "
        f"{total}"
    )

    print()

    print(
        f"Abstention accuracy:       "
        f"{abstention_accuracy * 100:.1f}%"
    )

    print(
        f"Supported recall:           "
        f"{supported_recall * 100:.1f}%"
    )

    print(
        f"Unsupported specificity:    "
        f"{unsupported_specificity * 100:.1f}%"
    )

    print(
        f"Precision when supported:   "
        f"{precision_when_supported * 100:.1f}%"
    )

    print(
        f"False-answer rate:           "
        f"{false_answer_rate * 100:.1f}%"
    )

    print(
        f"False-abstention rate:       "
        f"{false_abstention_rate * 100:.1f}%"
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

        for item in failures:

            print(
                f"\n- {item['id']}"
            )

            print(
                f"  {item['question']}"
            )

            print(
                "  Expected supported: "
                f"{item['expected_supported']}"
            )

            print(
                "  Predicted supported: "
                f"{item['predicted_supported']}"
            )

            print(
                "  Failure type: "
                f"{item['failure_type']}"
            )

            print(
                "  First gold source rank: "
                f"{item['first_gold_source_rank']}"
            )

            print(
                "  Reason: "
                f"{item['reason']}"
            )

            print(
                "  Evidence sources:"
            )

            for source_id in (
                item[
                    "evidence_source_ids"
                ]
            ):

                print(
                    f"    - {source_id}"
                )

    print(
        "\nNOTE:"
    )

    print(
        "Gold-source rank is only a diagnostic. "
        "A gold source in the top 3 does not prove "
        "that the exact answer-bearing passage was "
        "retrieved."
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