from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

from src.retrieval.search_with_reranker import (
    EMBEDDING_MODEL_NAME,
    understand_query,
    retrieve_candidates,
    load_local_corpus,
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
    / "data/evaluation/frozen_retrieval_metrics.json"
)


# ==================================================
# CONFIG
# ==================================================

# We evaluate deeper than K=5 so that failures
# can still show where the first relevant source
# appeared.
EVAL_TOP_K = 10


# ==================================================
# HELPERS
# ==================================================


def load_gold_dataset():

    data = json.loads(
        GOLD_PATH.read_text(
            encoding="utf-8"
        )
    )

    # Support either:
    #
    # [...]
    #
    # or:
    #
    # {"questions": [...]}
    #
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
            "Gold dataset must be a JSON list "
            "or contain a 'questions' list."
        )

    return data


def get_source_id(result):

    chunk = result.get(
        "chunk",
        {}
    )

    return chunk.get(
        "source_id"
    )


def validate_gold_dataset(
    questions,
    chunks,
):

    corpus_source_ids = {
        chunk.get("source_id")
        for chunk in chunks
        if chunk.get("source_id")
    }

    seen_ids = set()

    problems = []

    answerable_count = 0
    unanswerable_count = 0

    for item in questions:

        question_id = item.get(
            "id"
        )

        if not question_id:

            problems.append(
                "Question missing ID."
            )

            continue

        if question_id in seen_ids:

            problems.append(
                f"Duplicate question ID: "
                f"{question_id}"
            )

        seen_ids.add(
            question_id
        )

        answerable = item.get(
            "answerable",
            True,
        )

        if answerable:

            answerable_count += 1

            expected_sources = (
                item.get(
                    "expected_source_ids",
                    []
                )
            )

            if not expected_sources:

                problems.append(
                    f"{question_id}: "
                    "answerable question has no "
                    "expected_source_ids."
                )

                continue

            for source_id in expected_sources:

                if (
                    source_id
                    not in corpus_source_ids
                ):

                    problems.append(
                        f"{question_id}: "
                        f"gold source '{source_id}' "
                        "does not exist in corpus."
                    )

        else:

            unanswerable_count += 1

    if problems:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "GOLD DATASET VALIDATION FAILED"
        )

        print(
            "=" * 60
        )

        for problem in problems:

            print(
                f"- {problem}"
            )

        raise ValueError(
            "Fix gold dataset validation "
            "errors before evaluation."
        )

    return {
        "total": len(
            questions
        ),
        "answerable": (
            answerable_count
        ),
        "unanswerable": (
            unanswerable_count
        ),
    }


# ==================================================
# METRICS
# ==================================================


def first_relevant_rank(
    source_ids,
    expected_sources,
):

    expected = set(
        expected_sources
    )

    for index, source_id in enumerate(
        source_ids,
        start=1,
    ):

        if source_id in expected:

            return index

    return None


def hit_at_k(
    source_ids,
    expected_sources,
    k,
):

    expected = set(
        expected_sources
    )

    return int(
        any(
            source_id in expected
            for source_id
            in source_ids[:k]
        )
    )


# ==================================================
# EVALUATION
# ==================================================


def evaluate():

    print(
        "Loading gold dataset..."
    )

    questions = (
        load_gold_dataset()
    )

    print(
        "Loading frozen corpus..."
    )

    chunks, embeddings = (
        load_local_corpus()
    )

    dataset_stats = (
        validate_gold_dataset(
            questions,
            chunks,
        )
    )

    print(
        "\nDataset:"
    )

    print(
        f"  Total:        "
        f"{dataset_stats['total']}"
    )

    print(
        f"  Answerable:   "
        f"{dataset_stats['answerable']}"
    )

    print(
        f"  Unanswerable: "
        f"{dataset_stats['unanswerable']}"
    )

    print(
        "\nLoading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    answerable_questions = [
        item
        for item in questions
        if item.get(
            "answerable",
            True,
        )
    ]

    results = []

    recall_1_total = 0
    recall_3_total = 0
    recall_5_total = 0

    reciprocal_rank_total = 0.0

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FROZEN RETRIEVAL EVALUATION"
    )

    print(
        "=" * 60
    )

    for number, item in enumerate(
        answerable_questions,
        start=1,
    ):

        question_id = item[
            "id"
        ]

        question = item[
            "question"
        ]

        expected_sources = (
            item[
                "expected_source_ids"
            ]
        )

        query_info = (
            understand_query(
                question
            )
        )

        candidates = (
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
                top_k=EVAL_TOP_K,
            )
        )

        retrieved_source_ids = [
            get_source_id(
                result
            )
            for result in candidates
        ]

        rank = (
            first_relevant_rank(
                retrieved_source_ids,
                expected_sources,
            )
        )

        recall_1 = hit_at_k(
            retrieved_source_ids,
            expected_sources,
            1,
        )

        recall_3 = hit_at_k(
            retrieved_source_ids,
            expected_sources,
            3,
        )

        recall_5 = hit_at_k(
            retrieved_source_ids,
            expected_sources,
            5,
        )

        reciprocal_rank = (
            1.0 / rank
            if rank is not None
            else 0.0
        )

        recall_1_total += (
            recall_1
        )

        recall_3_total += (
            recall_3
        )

        recall_5_total += (
            recall_5
        )

        reciprocal_rank_total += (
            reciprocal_rank
        )

        result = {

            "id": question_id,

            "question": (
                question
            ),

            "topic_filter": (
                query_info[
                    "topic_filter"
                ]
            ),

            "retrieval_query": (
                query_info[
                    "retrieval_query"
                ]
            ),

            "expected_source_ids": (
                expected_sources
            ),

            "retrieved_source_ids": (
                retrieved_source_ids
            ),

            "recall_at_1": (
                recall_1
            ),

            "recall_at_3": (
                recall_3
            ),

            "recall_at_5": (
                recall_5
            ),

            "first_relevant_rank": (
                rank
            ),

            "reciprocal_rank": (
                reciprocal_rank
            ),
        }

        results.append(
            result
        )

        print(
            f"\n[{number}/"
            f"{len(answerable_questions)}] "
            f"{question_id}"
        )

        print(
            question
        )

        print(
            "Gold:"
        )

        for source_id in expected_sources:

            print(
                f"  - {source_id}"
            )

        print(
            f"Topic filter: "
            f"{query_info['topic_filter']}"
        )

        print(
            f"Recall@1: "
            f"{recall_1}"
        )

        print(
            f"Recall@3: "
            f"{recall_3}"
        )

        print(
            f"Recall@5: "
            f"{recall_5}"
        )

        print(
            f"First relevant rank: "
            f"{rank}"
        )

        if recall_5 == 0:

            print(
                "Top 5 retrieved sources:"
            )

            for rank_number, source_id in enumerate(
                retrieved_source_ids[:5],
                start=1,
            ):

                print(
                    f"  {rank_number}. "
                    f"{source_id}"
                )

    # ==================================================
    # AGGREGATE
    # ==================================================

    total = len(
        answerable_questions
    )

    recall_at_1 = (
        recall_1_total
        / total
        if total
        else 0.0
    )

    recall_at_3 = (
        recall_3_total
        / total
        if total
        else 0.0
    )

    recall_at_5 = (
        recall_5_total
        / total
        if total
        else 0.0
    )

    mrr = (
        reciprocal_rank_total
        / total
        if total
        else 0.0
    )

    failed_at_5 = [
        result
        for result in results
        if result[
            "recall_at_5"
        ] == 0
    ]

    metrics = {

        "dataset": {
            "total_questions": (
                dataset_stats[
                    "total"
                ]
            ),
            "answerable_questions": (
                dataset_stats[
                    "answerable"
                ]
            ),
            "unanswerable_questions": (
                dataset_stats[
                    "unanswerable"
                ]
            ),
        },

        "retrieval_config": {
            "reranker_enabled": (
                False
            ),
            "source_diversity": (
                True
            ),
            "max_chunks_per_source": (
                2
            ),
            "evaluation_depth": (
                EVAL_TOP_K
            ),
        },

        "metrics": {
            "recall_at_1": (
                recall_at_1
            ),
            "recall_at_3": (
                recall_at_3
            ),
            "recall_at_5": (
                recall_at_5
            ),
            "mrr": (
                mrr
            ),
            "failures_at_5": (
                len(
                    failed_at_5
                )
            ),
        },

        "results": (
            results
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            metrics,
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
        "FROZEN RETRIEVAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Answerable questions: "
        f"{total}"
    )

    print(
        "\nPRODUCTION RETRIEVAL"
    )

    print(
        f"Recall@1: "
        f"{recall_at_1 * 100:.1f}%"
    )

    print(
        f"Recall@3: "
        f"{recall_at_3 * 100:.1f}%"
    )

    print(
        f"Recall@5: "
        f"{recall_at_5 * 100:.1f}%"
    )

    print(
        f"MRR:      "
        f"{mrr:.3f}"
    )

    print(
        f"\nRecall@5 failures: "
        f"{len(failed_at_5)}"
    )

    if failed_at_5:

        print(
            "\nFAILED QUESTIONS:"
        )

        for result in failed_at_5:

            print(
                f"  - {result['id']}: "
                f"{result['question']}"
            )

            print(
                "    First relevant rank: "
                f"{result['first_relevant_rank']}"
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