from pathlib import Path
import json
import re
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


# ==================================================
# PROJECT IMPORTS
# ==================================================

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
    / "data/evaluation/questions.json"
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
    / "data/evaluation/retrieval_report.json"
)


# ==================================================
# MODELS
# ==================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==================================================
# EVALUATION SETTINGS
# ==================================================

# Experimental legacy retrieval threshold.
#
# We already know this is not good enough to act
# as our final production answerability gate.
#
# Keep it here only so we can compare experiments
# consistently with our previous baseline.
RETRIEVAL_THRESHOLD = 0.35

TOPIC_EVAL_K = 3


# ==================================================
# TOPIC MATCHING
# ==================================================


def normalize_topic(
    value,
):
    """
    Normalize topic metadata into tokens.

    Example:

        "H-1B grace period"

    becomes roughly:

        {"h", "1b", "grace", "period"}

    This allows:

        expected:
        H-1B grace period

        actual:
        H-1B layoff termination grace period

    to correctly count as a topic match.
    """

    if not value:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        str(value).lower(),
    )

    return set(
        tokens
    )


def topic_matches(
    expected_topic,
    actual_topic,
):
    """
    Evaluation-only topic matcher.

    The expected topic tokens must all exist in
    the actual retrieved topic metadata.

    This avoids the old brittle substring test.
    """

    expected_tokens = normalize_topic(
        expected_topic
    )

    actual_tokens = normalize_topic(
        actual_topic
    )

    if not expected_tokens:
        return False

    return expected_tokens.issubset(
        actual_tokens
    )


# ==================================================
# RESULT SERIALIZATION
# ==================================================


def serialize_result(
    result,
):
    """
    Convert a retrieval/reranker result into a
    JSON-safe flattened structure.

    Keeping these fields in the evaluation report
    makes later confidence and authority diagnostics
    much easier.
    """

    chunk = result.get(
        "chunk",
        {},
    )

    return {
        "chunk_index": result.get(
            "chunk_index"
        ),

        "chunk_id": chunk.get(
            "chunk_id"
        ),

        "source_id": chunk.get(
            "source_id"
        ),

        "agency": chunk.get(
            "agency"
        ),

        "document": chunk.get(
            "document"
        ),

        "stage": chunk.get(
            "stage"
        ),

        "topic": chunk.get(
            "topic"
        ),

        "url": chunk.get(
            "url"
        ),

        # ------------------------------------------
        # SOURCE POLICY
        # ------------------------------------------

        "authority_type": (
            result.get(
                "authority_type"
            )
            or chunk.get(
                "authority_type"
            )
        ),

        "authority_score": result.get(
            "authority_score"
        ),

        "authority_adjustment": result.get(
            "authority_adjustment"
        ),

        "freshness_score": result.get(
            "freshness_score"
        ),

        "freshness_adjustment": result.get(
            "freshness_adjustment"
        ),

        "source_status": (
            result.get(
                "source_status"
            )
            or chunk.get(
                "source_status"
            )
        ),

        "active": result.get(
            "active"
        ),

        # ------------------------------------------
        # RETRIEVAL SIGNALS
        # ------------------------------------------

        "retrieval_score": result.get(
            "retrieval_score"
        ),

        "reranker_score": result.get(
            "reranker_score"
        ),

        "final_score": result.get(
            "final_score",
            result.get(
                "reranker_score"
            ),
        ),

        # Useful for manual debugging.
        "text": chunk.get(
            "text",
            "",
        ),
    }


# ==================================================
# SINGLE QUESTION EVALUATION
# ==================================================


def evaluate_question(
    test_case,
    chunks,
    embeddings,
    embedding_model,
    reranker,
):
    question = test_case[
        "question"
    ]

    expected_answerable = bool(
        test_case[
            "answerable"
        ]
    )

    expected_topic = test_case.get(
        "topic"
    )

    # ----------------------------------------------
    # QUERY UNDERSTANDING
    # ----------------------------------------------

    query_info = understand_query(
        question
    )

    retrieval_query = query_info[
        "retrieval_query"
    ]

    topic_filter = query_info[
        "topic_filter"
    ]

    # ----------------------------------------------
    # RETRIEVAL
    # ----------------------------------------------

    candidates = retrieve_candidates(
        retrieval_query,
        chunks,
        embeddings,
        embedding_model,
        topic_filter,
    )

    if candidates:

        top_retrieval_score = max(
            candidate[
                "retrieval_score"
            ]
            for candidate in candidates
        )

    else:

        top_retrieval_score = (
            float("-inf")
        )

    # ----------------------------------------------
    # LEGACY ANSWERABILITY PREDICTION
    # ----------------------------------------------

    predicted_answerable = (
        top_retrieval_score
        >= RETRIEVAL_THRESHOLD
    )

    answerability_correct = (
        predicted_answerable
        == expected_answerable
    )

    # ----------------------------------------------
    # RERANKING
    # ----------------------------------------------

    if candidates:

        ranked_results = rerank_candidates(
            question,
            retrieval_query,
            candidates,
            reranker,
        )

    else:

        ranked_results = []

    serialized_results = [
        serialize_result(
            result
        )
        for result in ranked_results
    ]

    # ----------------------------------------------
    # TOPIC EVALUATION
    # ----------------------------------------------

    top1_topic_match = False
    top3_topic_hit = False

    if (
        expected_answerable
        and expected_topic
        and serialized_results
    ):

        top1_topic_match = topic_matches(
            expected_topic,
            serialized_results[
                0
            ].get(
                "topic"
            ),
        )

        top3_topic_hit = any(
            topic_matches(
                expected_topic,
                result.get(
                    "topic"
                ),
            )
            for result in serialized_results[
                :TOPIC_EVAL_K
            ]
        )

    # ----------------------------------------------
    # TOP RESULT DETAILS
    # ----------------------------------------------

    if serialized_results:

        top_result = (
            serialized_results[
                0
            ]
        )

        top_document = (
            top_result.get(
                "document"
            )
        )

        top_topic = (
            top_result.get(
                "topic"
            )
        )

        top_reranker_score = (
            top_result.get(
                "reranker_score"
            )
        )

        top_final_score = (
            top_result.get(
                "final_score"
            )
        )

        top_authority_type = (
            top_result.get(
                "authority_type"
            )
        )

    else:

        top_document = None
        top_topic = None
        top_reranker_score = None
        top_final_score = None
        top_authority_type = None

    return {
        "id": test_case[
            "id"
        ],

        "question": question,

        "stage": test_case.get(
            "stage"
        ),

        "expected_topic": (
            expected_topic
        ),

        "expected_answerable": (
            expected_answerable
        ),

        "predicted_answerable": (
            predicted_answerable
        ),

        "answerability_correct": (
            answerability_correct
        ),

        "top_retrieval_score": (
            float(
                top_retrieval_score
            )
            if candidates
            else None
        ),

        "top_reranker_score": (
            top_reranker_score
        ),

        "top_final_score": (
            top_final_score
        ),

        "top_document": (
            top_document
        ),

        "top_topic": (
            top_topic
        ),

        "top_authority_type": (
            top_authority_type
        ),

        "top1_topic_match": (
            top1_topic_match
        ),

        "top3_topic_hit": (
            top3_topic_hit
        ),

        "retrieval_query": (
            retrieval_query
        ),

        "topic_filter": (
            topic_filter
        ),

        "results": (
            serialized_results
        ),
    }


# ==================================================
# SUMMARY METRICS
# ==================================================


def calculate_summary(
    results,
):
    total = len(
        results
    )

    answerable_results = [
        result
        for result in results
        if result[
            "expected_answerable"
        ]
    ]

    unanswerable_results = [
        result
        for result in results
        if not result[
            "expected_answerable"
        ]
    ]

    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    true_positive = sum(
        1
        for result in results
        if (
            result[
                "expected_answerable"
            ]
            and result[
                "predicted_answerable"
            ]
        )
    )

    true_negative = sum(
        1
        for result in results
        if (
            not result[
                "expected_answerable"
            ]
            and not result[
                "predicted_answerable"
            ]
        )
    )

    false_positive = sum(
        1
        for result in results
        if (
            not result[
                "expected_answerable"
            ]
            and result[
                "predicted_answerable"
            ]
        )
    )

    false_negative = sum(
        1
        for result in results
        if (
            result[
                "expected_answerable"
            ]
            and not result[
                "predicted_answerable"
            ]
        )
    )

    correct = (
        true_positive
        + true_negative
    )

    # ----------------------------------------------
    # ANSWERABILITY METRICS
    # ----------------------------------------------

    accuracy = (
        correct / total
        if total
        else 0.0
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

    # ----------------------------------------------
    # TOPIC METRICS
    # ----------------------------------------------

    top1_topic_correct = sum(
        1
        for result in answerable_results
        if result[
            "top1_topic_match"
        ]
    )

    top3_topic_correct = sum(
        1
        for result in answerable_results
        if result[
            "top3_topic_hit"
        ]
    )

    top1_topic_accuracy = (
        top1_topic_correct
        / len(
            answerable_results
        )
        if answerable_results
        else 0.0
    )

    top3_topic_accuracy = (
        top3_topic_correct
        / len(
            answerable_results
        )
        if answerable_results
        else 0.0
    )

    return {
        "total_questions": (
            total
        ),

        "answerable_questions": (
            len(
                answerable_results
            )
        ),

        "unanswerable_questions": (
            len(
                unanswerable_results
            )
        ),

        "correct": correct,

        "accuracy": accuracy,

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

        "top1_topic_correct": (
            top1_topic_correct
        ),

        "top1_topic_accuracy": (
            top1_topic_accuracy
        ),

        "top3_topic_correct": (
            top3_topic_correct
        ),

        "top3_topic_accuracy": (
            top3_topic_accuracy
        ),

        "retrieval_threshold": (
            RETRIEVAL_THRESHOLD
        ),
    }


# ==================================================
# PRINT QUESTION RESULT
# ==================================================


def print_question_result(
    number,
    total,
    result,
):
    print()

    print(
        "----------------------------------------"
    )

    print(
        f"Question {number}/{total}"
    )

    print()

    status = (
        "PASS"
        if result[
            "answerability_correct"
        ]
        else "FAIL"
    )

    print(
        f"[{status}] "
        f"{result['id']}"
    )

    print(
        f"Question: "
        f"{result['question']}"
    )

    print(
        f"Expected answerable: "
        f"{result['expected_answerable']}"
    )

    print(
        f"Predicted answerable: "
        f"{result['predicted_answerable']}"
    )

    score = result.get(
        "top_retrieval_score"
    )

    if score is not None:

        print(
            f"Top retrieval score: "
            f"{score:.4f}"
        )

    if result[
        "expected_answerable"
    ]:

        print(
            f"Expected topic: "
            f"{result['expected_topic']}"
        )

        print(
            f"Top-1 topic match: "
            f"{result['top1_topic_match']}"
        )

        print(
            f"Top-3 topic hit: "
            f"{result['top3_topic_hit']}"
        )

    if result.get(
        "top_document"
    ):

        print(
            f"Top document: "
            f"{result['top_document']}"
        )

        print(
            f"Top topic: "
            f"{result['top_topic']}"
        )

        print(
            f"Authority: "
            f"{result['top_authority_type']}"
        )

    reranker_score = result.get(
        "top_reranker_score"
    )

    if reranker_score is not None:

        print(
            f"Reranker score: "
            f"{reranker_score:.4f}"
        )

    final_score = result.get(
        "top_final_score"
    )

    if final_score is not None:

        print(
            f"Final score: "
            f"{final_score:.4f}"
        )


# ==================================================
# PRINT SUMMARY
# ==================================================


def print_summary(
    summary,
):
    print()

    print(
        "========================================"
    )

    print(
        "KK-GPT RETRIEVAL EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Total questions: "
        f"{summary['total_questions']}"
    )

    print(
        f"Answerable questions: "
        f"{summary['answerable_questions']}"
    )

    print(
        f"Unanswerable questions: "
        f"{summary['unanswerable_questions']}"
    )

    print()

    print(
        "ANSWERABILITY"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Correct: "
        f"{summary['correct']}"
        f"/{summary['total_questions']}"
    )

    print(
        f"Accuracy: "
        f"{summary['accuracy']:.1%}"
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
        "RETRIEVAL / RERANKING"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Top-1 topic correct: "
        f"{summary['top1_topic_correct']}"
        f"/{summary['answerable_questions']}"
    )

    print(
        f"Top-1 topic accuracy: "
        f"{summary['top1_topic_accuracy']:.1%}"
    )

    print(
        f"Top-3 topic correct: "
        f"{summary['top3_topic_correct']}"
        f"/{summary['answerable_questions']}"
    )

    print(
        f"Top-3 topic accuracy: "
        f"{summary['top3_topic_accuracy']:.1%}"
    )

    print()

    print(
        f"Current retrieval threshold: "
        f"{summary['retrieval_threshold']}"
    )

    print(
        "========================================"
    )


# ==================================================
# MAIN
# ==================================================


def main():

    # ----------------------------------------------
    # LOAD QUESTIONS
    # ----------------------------------------------

    print(
        "Loading evaluation questions..."
    )

    questions = json.loads(
        QUESTIONS_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Loaded "
        f"{len(questions)} questions."
    )

    # ----------------------------------------------
    # LOAD CORPUS
    # ----------------------------------------------

    print()

    print(
        "Loading corpus..."
    )

    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Corpus chunks: "
        f"{len(chunks)}"
    )

    print(
        f"Embedding rows: "
        f"{len(embeddings)}"
    )

    if len(
        chunks
    ) != len(
        embeddings
    ):

        raise ValueError(
            "Corpus chunks and embeddings do not "
            "match. Run embed_corpus.py first."
        )

    # ----------------------------------------------
    # MODELS
    # ----------------------------------------------

    print()

    print(
        "Loading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )

    print(
        "Loading reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    # ----------------------------------------------
    # EVALUATION
    # ----------------------------------------------

    print()

    print(
        "Running evaluation..."
    )

    results = []

    for number, test_case in enumerate(
        questions,
        start=1,
    ):

        result = evaluate_question(
            test_case,
            chunks,
            embeddings,
            embedding_model,
            reranker,
        )

        results.append(
            result
        )

        print_question_result(
            number,
            len(questions),
            result,
        )

    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    summary = calculate_summary(
        results
    )

    report = {
        "summary": summary,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
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