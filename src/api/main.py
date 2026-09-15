import json
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from src.retrieval.query_policy import classify_query

from src.retrieval.search_with_reranker import (
    EMBEDDING_MODEL_NAME,
    load_local_corpus,
    retrieve_candidates,
    understand_query,
)

from src.generation.generate_answer import (
    generate_grounded_answer,
)


# ==================================================
# CONFIG
# ==================================================

from src.core.settings import (
    RETRIEVAL_TOP_K,
)


# ==================================================
# LOGGING
# ==================================================

logger = logging.getLogger(
    "kkgpt.api"
)

logger.setLevel(
    logging.INFO
)

logger.propagate = False

if not logger.handlers:

    handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    logger.addHandler(
        handler
    )


def log_event(
    event: str,
    **fields,
):

    payload = {
        "event": event,
        **fields,
    }

    logger.info(
        json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
    )


# ==================================================
# APPLICATION STATE
# ==================================================

chunks = None
embeddings = None
embedding_model = None


# ==================================================
# REQUEST / RESPONSE MODELS
# ==================================================


class AskRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    question: str = Field(
        min_length=2,
        max_length=2000,
    )


class SourceResponse(BaseModel):

    citation_id: int

    source_id: Optional[str] = None

    agency: Optional[str] = None

    document: Optional[str] = None

    topic: Optional[str] = None

    url: Optional[str] = None

    authority_type: Optional[str] = None


class TimingResponse(BaseModel):

    query_policy_ms: float

    retrieval_ms: float

    generation_ms: float

    total_ms: float


class AskResponse(BaseModel):

    request_id: str

    question: str

    category: str

    routed_to_rag: bool

    answer: str

    abstained: bool

    latency_ms: float

    timings_ms: TimingResponse

    sources: list[SourceResponse]


class HealthResponse(BaseModel):

    status: str

    corpus_loaded: bool

    embedding_model_loaded: bool

    corpus_chunks: int

    embedding_model: str


# ==================================================
# STARTUP
# ==================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global chunks
    global embeddings
    global embedding_model

    startup_started = perf_counter()

    log_event(
        "startup_started"
    )

    chunks, embeddings = (
        load_local_corpus()
    )

    log_event(
        "corpus_loaded",
        corpus_chunks=len(chunks),
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    startup_ms = (
        perf_counter()
        - startup_started
    ) * 1000

    log_event(
        "startup_complete",
        corpus_chunks=len(chunks),
        embedding_model=(
            EMBEDDING_MODEL_NAME
        ),
        startup_ms=round(
            startup_ms,
            2,
        ),
    )

    yield

    log_event(
        "shutdown"
    )


# ==================================================
# APP
# ==================================================


app = FastAPI(
    title="KK-GPT API",
    description=(
        "Official-source U.S. immigration "
        "RAG API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==================================================
# HEALTH
# ==================================================


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    corpus_loaded = (
        chunks is not None
        and embeddings is not None
    )

    model_loaded = (
        embedding_model
        is not None
    )

    return HealthResponse(
        status=(
            "ok"
            if corpus_loaded
            and model_loaded
            else "not_ready"
        ),

        corpus_loaded=(
            corpus_loaded
        ),

        embedding_model_loaded=(
            model_loaded
        ),

        corpus_chunks=(
            len(chunks)
            if chunks is not None
            else 0
        ),

        embedding_model=(
            EMBEDDING_MODEL_NAME
        ),
    )


# ==================================================
# ASK
# ==================================================


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
):

    request_id = str(
        uuid4()
    )

    total_started = (
        perf_counter()
    )

    question = (
        request.question.strip()
    )

    log_event(
        "request_started",
        request_id=request_id,
        endpoint="/ask",
        question_length=len(
            question
        ),
    )

    if (
        chunks is None
        or embeddings is None
        or embedding_model is None
    ):

        log_event(
            "request_failed",
            request_id=request_id,
            stage="readiness",
            status_code=503,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "KK-GPT is not ready."
            ),
        )

    if not question:

        log_event(
            "request_failed",
            request_id=request_id,
            stage="validation",
            status_code=422,
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Question cannot be empty."
            ),
        )

    query_policy_ms = 0.0
    retrieval_ms = 0.0
    generation_ms = 0.0

    current_stage = (
        "query_policy"
    )

    try:

        # ==================================================
        # STEP 1 — QUERY POLICY
        # ==================================================

        stage_started = (
            perf_counter()
        )

        policy = classify_query(
            question
        )

        query_policy_ms = (
            perf_counter()
            - stage_started
        ) * 1000

        category = policy.get(
            "category",
            "UNKNOWN",
        )

        route_to_rag = (
            policy.get(
                "route_to_rag"
            )
            is True
        )

        log_event(
            "query_policy_complete",
            request_id=request_id,
            category=category,
            routed_to_rag=(
                route_to_rag
            ),
            latency_ms=round(
                query_policy_ms,
                2,
            ),
        )

        # ==================================================
        # NON-RAG QUERY
        # ==================================================

        if not route_to_rag:

            total_ms = (
                perf_counter()
                - total_started
            ) * 1000

            log_event(
                "request_complete",
                request_id=request_id,
                category=category,
                routed_to_rag=False,
                abstained=True,
                query_policy_ms=round(
                    query_policy_ms,
                    2,
                ),
                retrieval_ms=0.0,
                generation_ms=0.0,
                total_ms=round(
                    total_ms,
                    2,
                ),
            )

            return AskResponse(
                request_id=request_id,

                question=question,

                category=category,

                routed_to_rag=False,

                answer=(
                    "This V1 answers official "
                    "U.S. immigration factual "
                    "questions from its official "
                    "source corpus. This question "
                    "is not routed to factual RAG."
                ),

                abstained=True,

                latency_ms=round(
                    total_ms,
                    2,
                ),

                timings_ms=TimingResponse(
                    query_policy_ms=round(
                        query_policy_ms,
                        2,
                    ),

                    retrieval_ms=0.0,

                    generation_ms=0.0,

                    total_ms=round(
                        total_ms,
                        2,
                    ),
                ),

                sources=[],
            )

        # ==================================================
        # STEP 2 — QUERY UNDERSTANDING
        # ==================================================

        current_stage = (
            "retrieval"
        )

        stage_started = (
            perf_counter()
        )

        query_info = (
            understand_query(
                question
            )
        )

        retrieval_query = (
            query_info[
                "retrieval_query"
            ]
        )

        topic_filter = (
            query_info[
                "topic_filter"
            ]
        )

        # ==================================================
        # STEP 3 — FROZEN RETRIEVAL
        # ==================================================

        results = (
            retrieve_candidates(
                retrieval_query=(
                    retrieval_query
                ),

                chunks=chunks,

                embeddings=embeddings,

                embedding_model=(
                    embedding_model
                ),

                topic_filter=(
                    topic_filter
                ),

                top_k=(
                    RETRIEVAL_TOP_K
                ),
            )
        )

        retrieval_ms = (
            perf_counter()
            - stage_started
        ) * 1000

        log_event(
            "retrieval_complete",
            request_id=request_id,
            topic_filter=(
                topic_filter
            ),
            result_count=len(
                results
            ),
            latency_ms=round(
                retrieval_ms,
                2,
            ),
        )

        # ==================================================
        # STEP 4 — EVIDENCE + GENERATION
        # ==================================================

        current_stage = (
            "generation"
        )

        stage_started = (
            perf_counter()
        )

        generation = (
            generate_grounded_answer(
                question,
                results,
            )
        )

        generation_ms = (
            perf_counter()
            - stage_started
        ) * 1000

        log_event(
            "generation_complete",
            request_id=request_id,
            abstained=generation.get(
                "abstained",
                True,
            ),
            latency_ms=round(
                generation_ms,
                2,
            ),
        )

        # ==================================================
        # STEP 5 — SOURCES
        # ==================================================

        source_rows = []

        if not generation.get(
            "abstained",
            True,
        ):

            for source in generation.get(
                "sources",
                [],
            ):

                source_rows.append(
                    SourceResponse(
                        citation_id=source[
                            "citation_id"
                        ],

                        source_id=source.get(
                            "source_id"
                        ),

                        agency=source.get(
                            "agency"
                        ),

                        document=source.get(
                            "document"
                        ),

                        topic=source.get(
                            "topic"
                        ),

                        url=source.get(
                            "url"
                        ),

                        authority_type=(
                            source.get(
                                "authority_type"
                            )
                        ),
                    )
                )

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        total_ms = (
            perf_counter()
            - total_started
        ) * 1000

        abstained = (
            generation.get(
                "abstained",
                True,
            )
        )

        log_event(
            "request_complete",
            request_id=request_id,
            category=category,
            routed_to_rag=True,
            abstained=abstained,
            source_count=len(
                source_rows
            ),
            query_policy_ms=round(
                query_policy_ms,
                2,
            ),
            retrieval_ms=round(
                retrieval_ms,
                2,
            ),
            generation_ms=round(
                generation_ms,
                2,
            ),
            total_ms=round(
                total_ms,
                2,
            ),
        )

        return AskResponse(
            request_id=request_id,

            question=question,

            category=category,

            routed_to_rag=True,

            answer=generation[
                "answer"
            ],

            abstained=abstained,

            latency_ms=round(
                total_ms,
                2,
            ),

            timings_ms=TimingResponse(
                query_policy_ms=round(
                    query_policy_ms,
                    2,
                ),

                retrieval_ms=round(
                    retrieval_ms,
                    2,
                ),

                generation_ms=round(
                    generation_ms,
                    2,
                ),

                total_ms=round(
                    total_ms,
                    2,
                ),
            ),

            sources=source_rows,
        )

    except HTTPException:

        raise

    except Exception as exc:

        total_ms = (
            perf_counter()
            - total_started
        ) * 1000

        log_event(
            "request_error",
            request_id=request_id,
            stage=current_stage,
            error_type=(
                type(exc).__name__
            ),
            total_ms=round(
                total_ms,
                2,
            ),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "KK-GPT failed to process "
                "the request."
            ),
        ) from exc