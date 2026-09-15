from pathlib import Path
import json
import sys

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)


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


from src.retrieval.search_with_reranker import (
    understand_query,
    retrieve_candidates,
    rerank_candidates,
)


# ==================================================
# PATHS
# ==================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions_gold_draft.json"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data/processed/corpus_chunks.json"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data/processed/corpus_embeddings.npy"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/rag_v1_gold_metrics.json"
)


# ==================================================
# MODELS
# ==================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)


# ==================================================
# LOADERS
# ==================================================


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_data():

    questions = load_json(
        QUESTIONS_PATH
    )

    chunks = load_json(
        CHUNKS_PATH
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    if len(chunks) != len(
        embeddings
    ):
        raise ValueError(
            "Chunk count and embedding count "
            "do not match."
        )

    return (
        questions,
        chunks,
        embeddings,
    )


# ==================================================
# GOLD DATASET VALIDATION
# ==================================================


def validate_gold_dataset(
    questions,
    chunks,
):
    """
    Validate that every answerable question has
    source-level gold labels and that every gold
    source exists in the current corpus.
    """

    corpus_source_ids = {
        chunk.get(
            "source_id"
        )
        for chunk in chunks
        if chunk.get(
            "source_id"
        )
    }

    errors = []

    answerable_count = 0

    for item in questions:

        if not item.get(
            "answerable",
            True,
        ):
            continue

        answerable_count += 1

        question_id = item.get(
            "id",
            "UNKNOWN",
        )

        gold_sources = item.get(
            "expected_source_ids",
            [],
        )

        if not gold_sources:

            errors.append(
                f"{question_id}: "
                "missing expected_source_ids"
            )

            continue

        for source_id in gold_sources:

            if source_id not in corpus_source_ids:

                errors.append(
                    f"{question_id}: "
                    f"gold source not found "
                    f"in corpus: {source_id}"
                )

    if errors:

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

        for error in errors:

            print(
                f"- {error}"
            )

        raise ValueError(
            "Fix gold dataset errors before "
            "running evaluation."
        )

    print(
        "\nGold dataset validation: PASS"
    )

    print(
        f"Answerable gold questions: "
        f"{answerable_count}"
    )


# ==================================================
# GOLD RELEVANCE
# ==================================================


def get_gold_source_ids(
    question_item,
):

    source_ids = question_item.get(
        "expected_source_ids",
        [],
    )

    if isinstance(
        source_ids,
        str,
    ):
        source_ids = [
            source_ids
        ]

    return set(
        source_ids
    )


def result_is_relevant(
    result,
    question_item,
):
    """
    A result is relevant only when its source_id
    appears in the manually assigned gold labels.

    No topic-level fallback.
    """

    chunk = result.get(
        "chunk",
        {},
    )

    source_id = chunk.get(
        "source_id"
    )

    gold_source_ids = (
        get_gold_source_ids(
            question_item
        )
    )

    return (
        source_id
        in gold_source_ids
    )


# ==================================================
# METRICS
# ==================================================


def recall_at_k(
    ranked_results,
    question_item,
    k,
):
    """
    For this dataset, Recall@K means:

    Did at least one relevant gold source appear
    within the top K results?

    This is effectively hit-rate style retrieval
    recall for question-level evaluation.
    """

    for result in ranked_results[:k]:

        if result_is_relevant(
            result,
            question_item,
        ):
            return 1

    return 0


def reciprocal_rank(
    ranked_results,
    question_item,
):

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):

        if result_is_relevant(
            result,
            question_item,
        ):

            return (
                1.0
                / rank
            )

    return 0.0


def first_relevant_rank(
    ranked_results,
    question_item,
):

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):

        if result_is_relevant(
            result,
            question_item,
        ):

            return rank

    return None


# ==================================================
# SERIALIZATION
# ==================================================


def serialize_result(
    result,
    question_item,
):

    chunk = result.get(
        "chunk",
        {},
    )

    return {
        "source_id": chunk.get(
            "source_id"
        ),
        "document": chunk.get(
            "document"
        ),
        "agency": chunk.get(
            "agency"
        ),
        "topic": chunk.get(
            "topic"
        ),
        "is_gold_relevant": (
            result_is_relevant(
                result,
                question_item,
            )
        ),
        "retrieval_score": (
            result.get(
                "retrieval_score"
            )
        ),
        "reranker_score": (
            result.get(
                "reranker_score"
            )
        ),
        "final_score": (
            result.get(
                "final_score"
            )
        ),
        "authority_type": (
            result.get(
                "authority_type"
            )
            or chunk.get(
                "authority_type"
            )
        ),
        "source_status": (
            result.get(
                "source_status"
            )
            or chunk.get(
                "source_status"
            )
        ),
    }


# ==================================================
# MAIN EVALUATION
# ==================================================


def evaluate():

    (
        questions,
        chunks,
        embeddings,
    ) = load_data()

    # ----------------------------------------------
    # Validate labels before loading models
    # ----------------------------------------------

    validate_gold_dataset(
        questions,
        chunks,
    )

    # ----------------------------------------------
    # Answerable questions only
    # ----------------------------------------------

    answerable_questions = [
        item
        for item in questions
        if item.get(
            "answerable",
            True,
        )
    ]

    print(
        "\nLoading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    print(
        "Loading reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RAG V1 GOLD-SOURCE RETRIEVAL EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Total dataset questions: "
        f"{len(questions)}"
    )

    print(
        f"Answerable questions: "
        f"{len(answerable_questions)}"
    )

    print(
        f"Corpus chunks: "
        f"{len(chunks)}"
    )

    # ==================================================
    # COUNTERS
    # ==================================================

    retrieval_r1_total = 0
    retrieval_r3_total = 0
    retrieval_r5_total = 0
    retrieval_rr_total = 0.0

    reranked_r1_total = 0
    reranked_r3_total = 0
    reranked_r5_total = 0
    reranked_rr_total = 0.0

    details = []

    # ==================================================
    # QUESTION LOOP
    # ==================================================

    for number, item in enumerate(
        answerable_questions,
        start=1,
    ):

        question_id = item.get(
            "id"
        )

        question = item[
            "question"
        ]

        gold_sources = sorted(
            get_gold_source_ids(
                item
            )
        )

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"[{number}/"
            f"{len(answerable_questions)}] "
            f"{question_id}"
        )

        print(
            question
        )

        print(
            "Gold sources:"
        )

        for source_id in gold_sources:

            print(
                f"  - {source_id}"
            )

        # ==============================================
        # QUERY UNDERSTANDING
        # ==============================================

        query_info = understand_query(
            question
        )

        retrieval_query = query_info[
            "retrieval_query"
        ]

        topic_filter = query_info[
            "topic_filter"
        ]

        # ==============================================
        # RETRIEVAL
        # ==============================================

        candidates = retrieve_candidates(
            retrieval_query,
            chunks,
            embeddings,
            embedding_model,
            topic_filter,
        )

        # ==============================================
        # RERANKING
        # ==============================================

        reranked_results = (
            rerank_candidates(
                question,
                retrieval_query,
                candidates,
                reranker,
            )
        )

        # ==============================================
        # BEFORE RERANKING METRICS
        # ==============================================

        retrieval_r1 = recall_at_k(
            candidates,
            item,
            1,
        )

        retrieval_r3 = recall_at_k(
            candidates,
            item,
            3,
        )

        retrieval_r5 = recall_at_k(
            candidates,
            item,
            5,
        )

        retrieval_rr = (
            reciprocal_rank(
                candidates,
                item,
            )
        )

        retrieval_rank = (
            first_relevant_rank(
                candidates,
                item,
            )
        )

        retrieval_r1_total += (
            retrieval_r1
        )

        retrieval_r3_total += (
            retrieval_r3
        )

        retrieval_r5_total += (
            retrieval_r5
        )

        retrieval_rr_total += (
            retrieval_rr
        )

        # ==============================================
        # AFTER RERANKING METRICS
        # ==============================================

        reranked_r1 = recall_at_k(
            reranked_results,
            item,
            1,
        )

        reranked_r3 = recall_at_k(
            reranked_results,
            item,
            3,
        )

        reranked_r5 = recall_at_k(
            reranked_results,
            item,
            5,
        )

        reranked_rr = (
            reciprocal_rank(
                reranked_results,
                item,
            )
        )

        reranked_rank = (
            first_relevant_rank(
                reranked_results,
                item,
            )
        )

        reranked_r1_total += (
            reranked_r1
        )

        reranked_r3_total += (
            reranked_r3
        )

        reranked_r5_total += (
            reranked_r5
        )

        reranked_rr_total += (
            reranked_rr
        )

        # ==============================================
        # CONSOLE RESULTS
        # ==============================================

        print(
            "\nBefore reranking:"
        )

        print(
            f"  Recall@1: "
            f"{retrieval_r1}"
        )

        print(
            f"  Recall@3: "
            f"{retrieval_r3}"
        )

        print(
            f"  Recall@5: "
            f"{retrieval_r5}"
        )

        print(
            f"  First relevant rank: "
            f"{retrieval_rank}"
        )

        print(
            "\nAfter reranking:"
        )

        print(
            f"  Recall@1: "
            f"{reranked_r1}"
        )

        print(
            f"  Recall@3: "
            f"{reranked_r3}"
        )

        print(
            f"  Recall@5: "
            f"{reranked_r5}"
        )

        print(
            f"  First relevant rank: "
            f"{reranked_rank}"
        )

        # ==============================================
        # SAVE QUESTION DETAIL
        # ==============================================

        details.append(
            {
                "id": question_id,
                "question": question,
                "stage": item.get(
                    "stage"
                ),
                "topic": item.get(
                    "topic"
                ),
                "gold_source_ids": (
                    gold_sources
                ),
                "evaluation_notes": (
                    item.get(
                        "_evaluation_notes",
                        "",
                    )
                ),
                "retrieval_query": (
                    retrieval_query
                ),
                "topic_filter": (
                    topic_filter
                ),
                "before_reranking": {
                    "recall_at_1": (
                        retrieval_r1
                    ),
                    "recall_at_3": (
                        retrieval_r3
                    ),
                    "recall_at_5": (
                        retrieval_r5
                    ),
                    "reciprocal_rank": (
                        retrieval_rr
                    ),
                    "first_relevant_rank": (
                        retrieval_rank
                    ),
                    "top_5": [
                        serialize_result(
                            result,
                            item,
                        )
                        for result
                        in candidates[:5]
                    ],
                },
                "after_reranking": {
                    "recall_at_1": (
                        reranked_r1
                    ),
                    "recall_at_3": (
                        reranked_r3
                    ),
                    "recall_at_5": (
                        reranked_r5
                    ),
                    "reciprocal_rank": (
                        reranked_rr
                    ),
                    "first_relevant_rank": (
                        reranked_rank
                    ),
                    "top_5": [
                        serialize_result(
                            result,
                            item,
                        )
                        for result
                        in reranked_results[:5]
                    ],
                },
            }
        )

    # ==================================================
    # AGGREGATE METRICS
    # ==================================================

    total = len(
        answerable_questions
    )

    if total == 0:

        raise ValueError(
            "No answerable questions found."
        )

    before_metrics = {
        "recall_at_1": (
            retrieval_r1_total
            / total
        ),
        "recall_at_3": (
            retrieval_r3_total
            / total
        ),
        "recall_at_5": (
            retrieval_r5_total
            / total
        ),
        "mrr": (
            retrieval_rr_total
            / total
        ),
    }

    after_metrics = {
        "recall_at_1": (
            reranked_r1_total
            / total
        ),
        "recall_at_3": (
            reranked_r3_total
            / total
        ),
        "recall_at_5": (
            reranked_r5_total
            / total
        ),
        "mrr": (
            reranked_rr_total
            / total
        ),
    }

    # ==================================================
    # DELTAS
    # ==================================================

    deltas = {
        "recall_at_1": (
            after_metrics[
                "recall_at_1"
            ]
            - before_metrics[
                "recall_at_1"
            ]
        ),
        "recall_at_3": (
            after_metrics[
                "recall_at_3"
            ]
            - before_metrics[
                "recall_at_3"
            ]
        ),
        "recall_at_5": (
            after_metrics[
                "recall_at_5"
            ]
            - before_metrics[
                "recall_at_5"
            ]
        ),
        "mrr": (
            after_metrics[
                "mrr"
            ]
            - before_metrics[
                "mrr"
            ]
        ),
    }

    # ==================================================
    # REPORT
    # ==================================================

    report = {
        "evaluation_type": (
            "manual_gold_source_ids"
        ),
        "dataset": {
            "total_questions": (
                len(questions)
            ),
            "answerable_questions": (
                total
            ),
            "unanswerable_questions": (
                len(questions)
                - total
            ),
        },
        "before_reranking": (
            before_metrics
        ),
        "after_reranking": (
            after_metrics
        ),
        "delta_after_reranking": (
            deltas
        ),
        "details": (
            details
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
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
        "GOLD-SOURCE RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAnswerable questions: "
        f"{total}"
    )

    print(
        "\nBEFORE RERANKING"
    )

    print(
        "Recall@1: "
        f"{before_metrics['recall_at_1']:.1%}"
    )

    print(
        "Recall@3: "
        f"{before_metrics['recall_at_3']:.1%}"
    )

    print(
        "Recall@5: "
        f"{before_metrics['recall_at_5']:.1%}"
    )

    print(
        "MRR:      "
        f"{before_metrics['mrr']:.3f}"
    )

    print(
        "\nAFTER RERANKING"
    )

    print(
        "Recall@1: "
        f"{after_metrics['recall_at_1']:.1%}"
    )

    print(
        "Recall@3: "
        f"{after_metrics['recall_at_3']:.1%}"
    )

    print(
        "Recall@5: "
        f"{after_metrics['recall_at_5']:.1%}"
    )

    print(
        "MRR:      "
        f"{after_metrics['mrr']:.3f}"
    )

    print(
        "\nRERANKER DELTA"
    )

    print(
        "Recall@1: "
        f"{deltas['recall_at_1']:+.1%}"
    )

    print(
        "Recall@3: "
        f"{deltas['recall_at_3']:+.1%}"
    )

    print(
        "Recall@5: "
        f"{deltas['recall_at_5']:+.1%}"
    )

    print(
        "MRR:      "
        f"{deltas['mrr']:+.3f}"
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