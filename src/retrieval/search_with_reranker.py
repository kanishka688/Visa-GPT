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


from src.retrieval.source_policy import (
    get_source_policy_signals,
)


# ==================================================
# PATHS
# ==================================================

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data/processed/corpus_chunks.json"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data/processed/corpus_embeddings.npy"
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
# RETRIEVAL CONFIG
# ==================================================

TOP_K_RETRIEVAL = 20
TOP_K_RERANKED = 10

# IMPORTANT:
#
# Candidate discovery must NOT depend on the number
# of final results requested.
#
# Otherwise:
#
# top_k=5  -> one candidate universe
# top_k=10 -> another candidate universe
#
# and ranking metrics become unstable.
RAW_RETRIEVAL_POOL_SIZE = 60

# Hard diversity limit.
MAX_CHUNKS_PER_SOURCE = 2


# ==================================================
# SOURCE POLICY WEIGHTS
# ==================================================

AUTHORITY_WEIGHT = 0.15
FRESHNESS_WEIGHT = 0.10
RETRIEVAL_TIEBREAK_WEIGHT = 0.05
INACTIVE_SOURCE_PENALTY = 2.0


# ==================================================
# QUERY HELPERS
# ==================================================


def normalize_query_text(
    value,
):

    return (
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
    )


def contains_pattern(
    text,
    pattern,
):

    return (
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


# ==================================================
# QUERY UNDERSTANDING
# ==================================================


def understand_query(
    question,
):

    original_question = (
        question.strip()
    )

    normalized = (
        normalize_query_text(
            original_question
        )
    )

    retrieval_query = (
        original_question
    )

    topic_filter = None

    # ==================================================
    # DAY 1 CPT
    # ==================================================

    if (
        contains_pattern(
            normalized,
            r"\bday[\s-]*1\s+cpt\b",
        )
        or contains_pattern(
            normalized,
            r"\bday\s+one\s+cpt\b",
        )
    ):

        retrieval_query = (
            "curricular practical training "
            "graduate student immediate participation "
            "internship required by curriculum "
            "one academic year eligibility "
            "CPT authorization"
        )

        topic_filter = "CPT"

    # ==================================================
    # CAP-GAP
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bcap[\s-]*gap\b",
        )
    ):

        retrieval_query = (
            "F-1 student beneficiary H-1B cap subject "
            "petition requesting change of status "
            "extension of duration of status "
            "grant of employment authorization "
            "automatically extended until April 1 "
            "fiscal year timely filed "
            "nonfrivolous petition"
        )

        topic_filter = "F-1"

    # ==================================================
    # ETA FORM 9089
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\beta[\s-]*(?:form\s*)?9089\b",
        )
    ):

        retrieval_query = (
            "ETA Form 9089 "
            "Application for Permanent Employment "
            "Certification PERM labor certification "
            "Department of Labor employer application"
        )

        topic_filter = "PERM"

    # ==================================================
    # STEM OPT
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bstem\s+opt\b",
        )
        or (
            "stem optional practical training"
            in normalized
        )
    ):

        topic_filter = "STEM OPT"

    # ==================================================
    # CPT
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bcpt\b",
        )
        or (
            "curricular practical training"
            in normalized
        )
    ):

        topic_filter = "CPT"

    # ==================================================
    # H-4
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bh[\s-]*4\b",
        )
    ):

        topic_filter = "H-4"

    # ==================================================
    # H-1B
    #
    # Keep this before generic OPT matching.
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bh[\s-]*1b\b",
        )
    ):

        topic_filter = "H-1B"

    # ==================================================
    # OPT
    #
    # Word boundary is important:
    #
    # "options" must NOT match "OPT".
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bopt\b",
        )
        or (
            "optional practical training"
            in normalized
        )
    ):

        topic_filter = "OPT"

    # ==================================================
    # PREVAILING WAGE
    # ==================================================

    elif (
        "prevailing wage"
        in normalized
        or contains_pattern(
            normalized,
            r"\bpwd\b",
        )
    ):

        topic_filter = (
            "Prevailing Wage"
        )

    # ==================================================
    # I-140
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bi[\s-]*140\b",
        )
    ):

        topic_filter = "I-140"

    # ==================================================
    # PRIORITY DATE
    # ==================================================

    elif (
        "priority date"
        in normalized
    ):

        topic_filter = (
            "Priority Date"
        )

    # ==================================================
    # PERM
    #
    # Word boundary is critical:
    #
    # PERM != permanent residence
    # ==================================================

    elif (
        contains_pattern(
            normalized,
            r"\bperm\b",
        )
        or (
            "labor certification"
            in normalized
        )
        or (
            "permanent employment certification"
            in normalized
        )
    ):

        topic_filter = "PERM"

    return {
        "original_query": (
            original_question
        ),

        "retrieval_query": (
            retrieval_query
        ),

        "topic_filter": (
            topic_filter
        ),
    }


# ==================================================
# TOPIC FILTERING
# ==================================================


def normalize_topic_text(
    value,
):

    if value is None:
        return ""

    return (
        str(value)
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def chunk_matches_topic(
    chunk,
    topic_filter,
):

    if not topic_filter:
        return True

    topic = (
        normalize_topic_text(
            chunk.get(
                "topic",
                "",
            )
        )
    )

    stage = (
        normalize_topic_text(
            chunk.get(
                "stage",
                "",
            )
        )
    )

    document = (
        normalize_topic_text(
            chunk.get(
                "document",
                "",
            )
        )
    )

    combined = (
        f"{topic} "
        f"{stage} "
        f"{document}"
    )

    target = (
        normalize_topic_text(
            topic_filter
        )
    )

    aliases = {

        "f 1": [
            "f 1",
            "f1",
        ],

        "cpt": [
            "cpt",
            "curricular practical training",
        ],

        "opt": [
            "opt",
            "optional practical training",
        ],

        "stem opt": [
            "stem opt",
            "stem optional practical training",
        ],

        "h 1b": [
            "h 1b",
            "h1b",
        ],

        "h 4": [
            "h 4",
            "h4",
        ],

        "perm": [
            "perm",
            "labor certification",
            "permanent employment",
        ],

        "prevailing wage": [
            "prevailing wage",
            "pwd",
        ],

        "i 140": [
            "i 140",
            "i140",
            "employment based immigrant",
        ],

        "priority date": [
            "priority date",
        ],
    }

    target_aliases = (
        aliases.get(
            target,
            [target],
        )
    )

    return any(
        alias in combined
        for alias in target_aliases
    )


def get_allowed_indices(
    chunks,
    topic_filter,
):

    if not topic_filter:

        return list(
            range(
                len(chunks)
            )
        )

    allowed = [
        index
        for index, chunk
        in enumerate(chunks)
        if chunk_matches_topic(
            chunk,
            topic_filter,
        )
    ]

    # Fail open if metadata filtering itself
    # accidentally produces zero candidates.
    if not allowed:

        return list(
            range(
                len(chunks)
            )
        )

    return allowed


# ==================================================
# SOURCE DIVERSITY
# ==================================================


def get_source_id(
    result,
):

    chunk = result.get(
        "chunk",
        {},
    )

    return (
        chunk.get(
            "source_id"
        )
        or chunk.get(
            "id"
        )
        or "UNKNOWN_SOURCE"
    )


def diversify_by_source(
    ranked_results,
    top_k,
    max_per_source=MAX_CHUNKS_PER_SOURCE,
):
    """
    Hard source-diversity constraint.

    No source may contribute more than
    max_per_source chunks.

    Unlike the previous implementation,
    this function does NOT refill the result set
    with skipped duplicate chunks.

    Therefore MAX_CHUNKS_PER_SOURCE is now truly
    a maximum rather than a soft preference.
    """

    selected = []

    source_counts = {}

    for result in ranked_results:

        source_id = (
            get_source_id(
                result
            )
        )

        current_count = (
            source_counts.get(
                source_id,
                0,
            )
        )

        if (
            current_count
            >= max_per_source
        ):
            continue

        selected.append(
            result
        )

        source_counts[
            source_id
        ] = (
            current_count
            + 1
        )

        if (
            len(selected)
            >= top_k
        ):
            break

    return selected


# ==================================================
# SEMANTIC RETRIEVAL
# ==================================================


def cosine_scores(
    query_embedding,
    corpus_embeddings,
):

    return np.dot(
        corpus_embeddings,
        query_embedding,
    )


def retrieve_candidates(
    retrieval_query,
    chunks,
    embeddings,
    embedding_model,
    topic_filter=None,
    top_k=TOP_K_RETRIEVAL,
):

    if (
        len(chunks)
        != len(embeddings)
    ):

        raise ValueError(
            "Chunk count and embedding count "
            "do not match."
        )

    allowed_indices = (
        get_allowed_indices(
            chunks,
            topic_filter,
        )
    )

    query_embedding = (
        embedding_model.encode(
            retrieval_query,
            normalize_embeddings=True,
        )
    )

    allowed_embeddings = (
        embeddings[
            allowed_indices
        ]
    )

    scores = (
        cosine_scores(
            query_embedding,
            allowed_embeddings,
        )
    )

    # --------------------------------------------------
    # FIXED RAW CANDIDATE POOL
    #
    # The candidate universe no longer changes just
    # because the caller asks for top 5 vs top 10.
    # --------------------------------------------------

    pool_size = min(
        len(
            allowed_indices
        ),
        max(
            RAW_RETRIEVAL_POOL_SIZE,
            top_k,
        ),
    )

    ranked_positions = (
        np.argsort(
            scores
        )[::-1]
    )[
        :pool_size
    ]

    raw_results = []

    for position in (
        ranked_positions
    ):

        corpus_index = (
            allowed_indices[
                int(position)
            ]
        )

        chunk = (
            chunks[
                corpus_index
            ]
        )

        raw_results.append(
            {
                "chunk": (
                    chunk
                ),

                "corpus_index": (
                    corpus_index
                ),

                "retrieval_score": float(
                    scores[
                        int(position)
                    ]
                ),
            }
        )

    return (
        diversify_by_source(
            raw_results,
            top_k=top_k,
            max_per_source=(
                MAX_CHUNKS_PER_SOURCE
            ),
        )
    )


# ==================================================
# SOURCE POLICY SCORING
# ==================================================


def calculate_final_score(
    reranker_score,
    retrieval_score,
    policy_signals,
):

    authority_score = float(
        policy_signals.get(
            "authority_score",
            3,
        )
    )

    freshness_score = float(
        policy_signals.get(
            "freshness_score",
            0.5,
        )
    )

    active = bool(
        policy_signals.get(
            "active",
            True,
        )
    )

    authority_adjustment = (
        (
            authority_score
            - 3.0
        )
        * AUTHORITY_WEIGHT
    )

    freshness_adjustment = (
        (
            freshness_score
            - 0.5
        )
        * FRESHNESS_WEIGHT
    )

    retrieval_adjustment = (
        retrieval_score
        * RETRIEVAL_TIEBREAK_WEIGHT
    )

    inactive_penalty = (
        0.0
        if active
        else INACTIVE_SOURCE_PENALTY
    )

    final_score = (
        reranker_score
        + authority_adjustment
        + freshness_adjustment
        + retrieval_adjustment
        - inactive_penalty
    )

    return {

        "final_score": float(
            final_score
        ),

        "authority_adjustment": float(
            authority_adjustment
        ),

        "freshness_adjustment": float(
            freshness_adjustment
        ),

        "retrieval_adjustment": float(
            retrieval_adjustment
        ),

        "inactive_penalty": float(
            inactive_penalty
        ),
    }


# ==================================================
# RERANKING
# ==================================================


def rerank_candidates(
    question,
    retrieval_query,
    candidates,
    reranker,
    top_k=TOP_K_RERANKED,
):

    if not candidates:
        return []

    pairs = []

    for candidate in (
        candidates
    ):

        chunk = (
            candidate[
                "chunk"
            ]
        )

        pairs.append(
            [
                question,
                chunk.get(
                    "text",
                    "",
                ),
            ]
        )

    reranker_scores = (
        reranker.predict(
            pairs
        )
    )

    reranked = []

    for candidate, raw_score in zip(
        candidates,
        reranker_scores,
    ):

        chunk = (
            candidate[
                "chunk"
            ]
        )

        retrieval_score = float(
            candidate.get(
                "retrieval_score",
                0.0,
            )
        )

        reranker_score = float(
            raw_score
        )

        policy_signals = (
            get_source_policy_signals(
                chunk
            )
        )

        score_components = (
            calculate_final_score(
                reranker_score=(
                    reranker_score
                ),
                retrieval_score=(
                    retrieval_score
                ),
                policy_signals=(
                    policy_signals
                ),
            )
        )

        reranked.append(
            {
                **candidate,

                "reranker_score": (
                    reranker_score
                ),

                "authority_type": (
                    policy_signals.get(
                        "authority_type"
                    )
                ),

                "authority_score": (
                    policy_signals.get(
                        "authority_score"
                    )
                ),

                "authority_adjustment": (
                    score_components[
                        "authority_adjustment"
                    ]
                ),

                "freshness_score": (
                    policy_signals.get(
                        "freshness_score"
                    )
                ),

                "freshness_adjustment": (
                    score_components[
                        "freshness_adjustment"
                    ]
                ),

                "retrieval_adjustment": (
                    score_components[
                        "retrieval_adjustment"
                    ]
                ),

                "source_status": (
                    policy_signals.get(
                        "source_status"
                    )
                ),

                "active": (
                    policy_signals.get(
                        "active"
                    )
                ),

                "publication_date": (
                    policy_signals.get(
                        "publication_date"
                    )
                ),

                "effective_date": (
                    policy_signals.get(
                        "effective_date"
                    )
                ),

                "last_reviewed_date": (
                    policy_signals.get(
                        "last_reviewed_date"
                    )
                ),

                "supersedes": (
                    policy_signals.get(
                        "supersedes"
                    )
                ),

                "superseded_by": (
                    policy_signals.get(
                        "superseded_by"
                    )
                ),

                "inactive_penalty": (
                    score_components[
                        "inactive_penalty"
                    ]
                ),

                "final_score": (
                    score_components[
                        "final_score"
                    ]
                ),
            }
        )

    reranked.sort(
        key=lambda item: (
            item[
                "final_score"
            ]
        ),
        reverse=True,
    )

    return (
        reranked[
            :top_k
        ]
    )


# ==================================================
# COMPLETE SEARCH
# ==================================================


def search_with_reranker(
    question,
    chunks,
    embeddings,
    embedding_model,
    reranker,
    top_k=TOP_K_RERANKED,
):

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
        )
    )

    reranked = (
        rerank_candidates(
            question=question,
            retrieval_query=(
                query_info[
                    "retrieval_query"
                ]
            ),
            candidates=(
                candidates
            ),
            reranker=reranker,
            top_k=top_k,
        )
    )

    return {

        "query_info": (
            query_info
        ),

        "candidates": (
            candidates
        ),

        "results": (
            reranked
        ),
    }


# ==================================================
# LOAD LOCAL CORPUS
# ==================================================


def load_local_corpus():

    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )

    embeddings = (
        np.load(
            EMBEDDINGS_PATH
        )
    )

    if (
        len(chunks)
        != len(embeddings)
    ):

        raise ValueError(
            "Chunk count and embedding count "
            "do not match."
        )

    return (
        chunks,
        embeddings,
    )


# ==================================================
# CLI TEST
# ==================================================


def main():

    print(
        "Loading corpus..."
    )

    chunks, embeddings = (
        load_local_corpus()
    )

    print(
        f"Chunks: {len(chunks)}"
    )

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

    reranker = (
        CrossEncoder(
            RERANKER_MODEL_NAME
        )
    )

    while True:

        print(
            "\n"
            + "=" * 70
        )

        question = input(
            "Question "
            "(or 'quit'): "
        ).strip()

        if question.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        if not question:
            continue

        search = (
            search_with_reranker(
                question,
                chunks,
                embeddings,
                embedding_model,
                reranker,
                top_k=5,
            )
        )

        query_info = (
            search[
                "query_info"
            ]
        )

        print(
            "\nQuery understanding:"
        )

        print(
            "Original: "
            f"{query_info['original_query']}"
        )

        print(
            "Retrieval: "
            f"{query_info['retrieval_query']}"
        )

        print(
            "Topic filter: "
            f"{query_info['topic_filter']}"
        )

        print(
            "\nTop results:"
        )

        for rank, result in enumerate(
            search[
                "results"
            ],
            start=1,
        ):

            chunk = (
                result[
                    "chunk"
                ]
            )

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"Rank {rank}"
            )

            print(
                "Source ID: "
                f"{chunk.get('source_id')}"
            )

            print(
                "Document: "
                f"{chunk.get('document')}"
            )

            print(
                "Topic: "
                f"{chunk.get('topic')}"
            )

            print(
                "Retrieval score: "
                f"{result['retrieval_score']:.4f}"
            )

            print(
                "Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            print(
                "Final score: "
                f"{result['final_score']:.4f}"
            )


# ==================================================
# ENTRY POINT
# ==================================================


if __name__ == "__main__":

    main()