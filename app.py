from pathlib import Path
import json

import numpy as np
import requests
import streamlit as st
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from src.retrieval.query_policy import (
    classify_query,
)

from src.retrieval.search_with_reranker import (
    understand_query,
    retrieve_candidates,
    rerank_candidates,
)

from src.generation.generate_answer import (
    generate_grounded_answer,
    ABSTAIN_MESSAGE,
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

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

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==================================================
# LOADERS
# ==================================================


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL
    )


@st.cache_data
def load_chunks():

    return json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8"
        )
    )


@st.cache_data
def load_embeddings():

    return np.load(
        EMBEDDINGS_PATH
    )


# ==================================================
# NON-RAG RESPONSES
# ==================================================


def get_policy_response(
    category,
):
    """
    Temporary V1 behavior for queries that should
    not enter official-source RAG.

    Recommendation/community/product workflows
    belong to later KK-Gpt versions.
    """

    responses = {

        "CASE_PREDICTION": (
            "I can explain the official rules and "
            "requirements that may apply, but I "
            "cannot predict whether USCIS, DOL, "
            "or the Department of State will "
            "approve or deny a specific case."
        ),

        "FUTURE_PREDICTION": (
            "I can explain current official rules "
            "and published information, but I "
            "cannot reliably predict future lottery "
            "results, Visa Bulletin movement, "
            "policy changes, or agency decisions."
        ),

        "RECOMMENDATION": (
            "This RAG V1 focuses on official "
            "immigration information. It does not "
            "currently rank or recommend specific "
            "schools, employers, attorneys, "
            "consultancies, or strategies."
        ),

        "PERSONAL_DECISION": (
            "I can explain the official immigration "
            "rules and consequences relevant to "
            "your options, but this RAG V1 does not "
            "make personal career or life decisions."
        ),

        "LOCAL_SERVICE": (
            "This RAG V1 focuses on official "
            "immigration information and does not "
            "currently search for local lawyers "
            "or service providers."
        ),

        "UNKNOWN": (
            "I could not confidently determine "
            "whether this question belongs in the "
            "official immigration RAG workflow."
        ),
    }

    return responses.get(
        category,
        responses["UNKNOWN"],
    )


# ==================================================
# PAGE CONFIG
# ==================================================


st.set_page_config(
    page_title="KK-GPT RAG V1",
    page_icon="🇺🇸",
    layout="centered",
)


st.title(
    "KK-GPT"
)

st.caption(
    "RAG V1 — Official U.S. immigration information "
    "from F-1 through I-140"
)


# ==================================================
# INPUT
# ==================================================


question = st.text_input(
    "Ask an immigration question",
    placeholder=(
        "Example: Does 12 months of full-time "
        "CPT affect OPT eligibility?"
    ),
)


ask_button = st.button(
    "Ask KK-GPT"
)


# ==================================================
# MAIN PIPELINE
# ==================================================


if ask_button and question.strip():

    question = question.strip()

    # ==============================================
    # STEP 1 — QUERY POLICY
    # ==============================================

    try:

        policy = classify_query(
            question
        )

    except requests.RequestException:

        st.error(
            "Could not connect to the local "
            "Ollama model."
        )

        st.stop()

    category = policy[
        "category"
    ]

    route_to_rag = policy[
        "route_to_rag"
    ]

    # ==============================================
    # STEP 2 — NON-RAG QUERY
    # ==============================================

    if not route_to_rag:

        st.subheader(
            "Answer"
        )

        st.write(
            get_policy_response(
                category
            )
        )

        with st.expander(
            "Developer details"
        ):

            st.write(
                f"Query category: {category}"
            )

            st.write(
                "Route to official RAG: False"
            )

        st.stop()

    # ==============================================
    # STEP 3 — LOAD CORPUS + MODELS
    # ==============================================

    chunks = load_chunks()

    embeddings = load_embeddings()

    embedding_model = (
        load_embedding_model()
    )

    reranker = (
        load_reranker()
    )

    # ==============================================
    # SAFETY CHECK
    # ==============================================

    if len(chunks) != len(
        embeddings
    ):

        st.error(
            "Corpus chunks and embeddings do not "
            "match. Run embed_corpus.py first."
        )

        st.stop()

    # ==============================================
    # STEP 4 — QUERY UNDERSTANDING
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
    # STEP 5 — RETRIEVAL
    # ==============================================

    candidates = retrieve_candidates(
        retrieval_query,
        chunks,
        embeddings,
        embedding_model,
        topic_filter,
    )

    if not candidates:

        st.subheader(
            "Answer"
        )

        st.write(
            ABSTAIN_MESSAGE
        )

        st.stop()

    # ==============================================
    # STEP 6 — RERANKING
    # ==============================================

    ranked_results = rerank_candidates(
        question,
        retrieval_query,
        candidates,
        reranker,
    )

    if not ranked_results:

        st.subheader(
            "Answer"
        )

        st.write(
            ABSTAIN_MESSAGE
        )

        st.stop()

    # ==============================================
    # STEP 7 — EVIDENCE GATE + GENERATION
    # ==============================================

    try:

        generation = (
            generate_grounded_answer(
                question,
                ranked_results,
            )
        )

    except requests.RequestException:

        st.error(
            "Could not connect to the local "
            "Ollama model during evidence "
            "checking or generation."
        )

        st.stop()

    answer = generation[
        "answer"
    ]

    abstained = generation[
        "abstained"
    ]

    sources = generation[
        "sources"
    ]

    evidence_check = generation[
        "evidence_check"
    ]

    citation_check = generation[
        "citation_check"
    ]

    # ==============================================
    # ANSWER
    # ==============================================

    st.subheader(
        "Answer"
    )

    if abstained:

        st.warning(
            answer
        )

    else:

        st.markdown(
            answer
        )

        st.caption(
            "Informational only. "
            "Not legal advice."
        )

    # ==============================================
    # SOURCES
    # ==============================================

    if (
        not abstained
        and sources
    ):

        st.subheader(
            "Official Sources"
        )

        for source in sources:

            citation_id = source[
                "citation_id"
            ]

            agency = (
                source.get(
                    "agency"
                )
                or "Official Source"
            )

            document = (
                source.get(
                    "document"
                )
                or "Official Document"
            )

            authority = (
                source.get(
                    "authority_type"
                )
                or "UNKNOWN"
            )

            url = source.get(
                "url"
            )

            st.markdown(
                f"**[{citation_id}] "
                f"{agency} — {document}**"
            )

            st.caption(
                f"Authority: {authority}"
            )

            if url:

                st.markdown(
                    f"[Open official source]"
                    f"({url})"
                )

    # ==============================================
    # DEVELOPER DEBUG
    # ==============================================

    with st.expander(
        "Developer pipeline details"
    ):

        st.write(
            f"Query category: "
            f"{category}"
        )

        st.write(
            "Route to official RAG: True"
        )

        st.write(
            f"Original query: "
            f"{question}"
        )

        st.write(
            f"Retrieval query: "
            f"{retrieval_query}"
        )

        st.write(
            f"Topic filter: "
            f"{topic_filter}"
        )

        st.write(
            f"Retrieved candidates: "
            f"{len(candidates)}"
        )

        st.write(
            f"Reranked results: "
            f"{len(ranked_results)}"
        )

        st.markdown(
            "### Evidence Gate"
        )

        st.write(
            f"Supported: "
            f"{evidence_check['supported']}"
        )

        st.write(
            f"Reason: "
            f"{evidence_check['reason']}"
        )

        st.markdown(
            "### Citation Validation"
        )

        if citation_check is None:

            st.write(
                "Not run because the system "
                "abstained before citation "
                "validation."
            )

        else:

            st.write(
                f"Valid: "
                f"{citation_check['valid']}"
            )

            st.write(
                f"Reason: "
                f"{citation_check['reason']}"
            )

            st.write(
                f"Citations used: "
                f"{citation_check['cited_ids']}"
            )

        st.markdown(
            "### Ranked Evidence"
        )

        for rank, result in enumerate(
            ranked_results,
            start=1,
        ):

            chunk = result[
                "chunk"
            ]

            st.markdown(
                f"#### Result {rank}"
            )

            st.write(
                f"Document: "
                f"{chunk.get('document')}"
            )

            st.write(
                f"Agency: "
                f"{chunk.get('agency')}"
            )

            st.write(
                f"Topic: "
                f"{chunk.get('topic')}"
            )

            st.write(
                f"Authority: "
                f"{result.get('authority_type')}"
            )

            st.write(
                f"Retrieval score: "
                f"{result['retrieval_score']:.4f}"
            )

            st.write(
                f"Reranker score: "
                f"{result['reranker_score']:.4f}"
            )

            st.write(
                f"Final score: "
                f"{result['final_score']:.4f}"
            )

            st.code(
                chunk.get(
                    "text",
                    "",
                )
            )