import json
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from src.buzz.web_buzz import (
    get_web_buzz,
)

from src.core.settings import (
    RETRIEVAL_TOP_K,
)

from src.generation.generate_answer import (
    generate_grounded_answer,
)

from src.retrieval.query_policy import (
    classify_query,
)

from src.retrieval.search_with_reranker import (
    EMBEDDING_MODEL_NAME,
    load_local_corpus,
    retrieve_candidates,
    understand_query,
)

from src.social.anlf_insight import (
    generate_anlf_insight,
)

from src.social.anlf_live_index import (
    ANLFLiveIndex,
)

from src.social.anlf_retrieval import (
    retrieve_anlf,
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

anlf_index = (
    ANLFLiveIndex()
)


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


class WebBuzzSourceResponse(BaseModel):

    title: str

    url: str

    domain: str

    source_type: str

    published_at: Optional[str] = None


class WebBuzzResponse(BaseModel):

    available: bool

    label: str = (
        "Current Web Buzz"
    )

    disclaimer: str = (
        "Community, news, forum, and web "
        "discussion only. This is not "
        "official immigration evidence."
    )

    summary: str

    source_class: str

    latency_ms: float

    sources: list[
        WebBuzzSourceResponse
    ]


class SocialSourceResponse(BaseModel):

    account: str

    platform: str

    topic: Optional[str] = None

    post_url: str

    posted_at: Optional[str] = None

    source_class: str


class SocialInsightResponse(BaseModel):

    account: str

    label: str

    available: bool

    disclaimer: str = (
        "Community/admin-reported information "
        "only. This is not official immigration "
        "guidance."
    )

    summary: str

    admin_points: list[str]

    community_points: list[str]

    caution: str

    source_class: str

    latency_ms: float

    sources: list[
        SocialSourceResponse
    ]


class TimingResponse(BaseModel):

    query_policy_ms: float

    retrieval_ms: float

    generation_ms: float

    web_buzz_ms: float

    social_ms: float

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

    sources: list[
        SourceResponse
    ]

    web_buzz: WebBuzzResponse

    social_insights: list[
        SocialInsightResponse
    ]


class HealthResponse(BaseModel):

    status: str

    corpus_loaded: bool

    embedding_model_loaded: bool

    corpus_chunks: int

    embedding_model: str

    social_index_loaded: bool

    social_chunks: int


# ==================================================
# EMPTY AUXILIARY RESPONSES
# ==================================================


def empty_web_buzz_response(
    latency_ms: float = 0.0,
) -> WebBuzzResponse:

    return WebBuzzResponse(
        available=False,

        summary="",

        source_class=(
            "WEB_BUZZ"
        ),

        latency_ms=round(
            latency_ms,
            2,
        ),

        sources=[],
    )


def empty_anlf_response(
    latency_ms: float = 0.0,
    caution: str = (
        "No sufficiently relevant America NRI "
        "Frustration content was found."
    ),
) -> SocialInsightResponse:

    return SocialInsightResponse(
        account=(
            "america_nri_la_frustration"
        ),

        label=(
            "America NRI Frustration"
        ),

        available=False,

        summary="",

        admin_points=[],

        community_points=[],

        caution=(
            caution
        ),

        source_class=(
            "SOCIAL_DISCUSSION"
        ),

        latency_ms=round(
            latency_ms,
            2,
        ),

        sources=[],
    )


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

    startup_started = (
        perf_counter()
    )

    log_event(
        "startup_started"
    )

    # ==================================================
    # OFFICIAL RAG
    # ==================================================

    chunks, embeddings = (
        load_local_corpus()
    )

    log_event(
        "corpus_loaded",

        corpus_chunks=len(
            chunks
        ),
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    )

    # ==================================================
    # ANLF SOCIAL INDEX
    # ==================================================
    #
    # Social data is optional.
    #
    # Failure to load it must NOT prevent
    # official KK-GPT from starting.
    # ==================================================

    social_loaded = (
        anlf_index.load_initial()
    )

    if social_loaded:

        log_event(
            "social_index_loaded",

            account=(
                "america_nri_la_frustration"
            ),

            chunk_count=(
                anlf_index.chunk_count()
            ),
        )

    else:

        log_event(
            "social_index_unavailable",

            account=(
                "america_nri_la_frustration"
            ),
        )

    # ==================================================
    # STARTUP COMPLETE
    # ==================================================

    startup_ms = (
        perf_counter()
        - startup_started
    ) * 1000

    log_event(
        "startup_complete",

        corpus_chunks=len(
            chunks
        ),

        embedding_model=(
            EMBEDDING_MODEL_NAME
        ),

        social_chunks=(
            anlf_index.chunk_count()
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
    title=(
        "KK-GPT API"
    ),

    description=(
        "Official-source U.S. immigration RAG "
        "with separate Web Buzz and curated "
        "community intelligence layers."
    ),

    version=(
        "1.2.0"
    ),

    lifespan=(
        lifespan
    ),
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

    social_loaded = (
        anlf_index.is_loaded()
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

        social_index_loaded=(
            social_loaded
        ),

        social_chunks=(
            anlf_index.chunk_count()
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

        request_id=(
            request_id
        ),

        endpoint=(
            "/ask"
        ),

        question_length=len(
            question
        ),
    )

    # ==================================================
    # READINESS
    # ==================================================

    if (
        chunks is None
        or embeddings is None
        or embedding_model is None
    ):

        log_event(
            "request_failed",

            request_id=(
                request_id
            ),

            stage=(
                "readiness"
            ),

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

            request_id=(
                request_id
            ),

            stage=(
                "validation"
            ),

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
    web_buzz_ms = 0.0
    social_ms = 0.0

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

        policy = (
            classify_query(
                question
            )
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

            request_id=(
                request_id
            ),

            category=(
                category
            ),

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

                request_id=(
                    request_id
                ),

                category=(
                    category
                ),

                routed_to_rag=False,

                abstained=True,

                query_policy_ms=round(
                    query_policy_ms,
                    2,
                ),

                retrieval_ms=0.0,

                generation_ms=0.0,

                web_buzz_ms=0.0,

                social_ms=0.0,

                total_ms=round(
                    total_ms,
                    2,
                ),
            )

            return AskResponse(
                request_id=(
                    request_id
                ),

                question=(
                    question
                ),

                category=(
                    category
                ),

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

                timings_ms=(
                    TimingResponse(
                        query_policy_ms=round(
                            query_policy_ms,
                            2,
                        ),

                        retrieval_ms=0.0,

                        generation_ms=0.0,

                        web_buzz_ms=0.0,

                        social_ms=0.0,

                        total_ms=round(
                            total_ms,
                            2,
                        ),
                    )
                ),

                sources=[],

                web_buzz=(
                    empty_web_buzz_response()
                ),

                social_insights=[
                    empty_anlf_response()
                ],
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
        # STEP 3 — OFFICIAL RETRIEVAL
        # ==================================================

        results = (
            retrieve_candidates(
                retrieval_query=(
                    retrieval_query
                ),

                chunks=(
                    chunks
                ),

                embeddings=(
                    embeddings
                ),

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

            request_id=(
                request_id
            ),

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
        # STEP 4 — OFFICIAL GENERATION
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

        abstained = (
            generation.get(
                "abstained",
                True,
            )
        )

        log_event(
            "generation_complete",

            request_id=(
                request_id
            ),

            abstained=(
                abstained
            ),

            latency_ms=round(
                generation_ms,
                2,
            ),
        )

        # ==================================================
        # OFFICIAL SOURCES
        # ==================================================

        source_rows = []

        if not abstained:

            for source in generation.get(
                "sources",
                [],
            ):

                source_rows.append(
                    SourceResponse(
                        citation_id=(
                            source[
                                "citation_id"
                            ]
                        ),

                        source_id=(
                            source.get(
                                "source_id"
                            )
                        ),

                        agency=(
                            source.get(
                                "agency"
                            )
                        ),

                        document=(
                            source.get(
                                "document"
                            )
                        ),

                        topic=(
                            source.get(
                                "topic"
                            )
                        ),

                        url=(
                            source.get(
                                "url"
                            )
                        ),

                        authority_type=(
                            source.get(
                                "authority_type"
                            )
                        ),
                    )
                )

        # ==================================================
        # STEP 5 — WEB BUZZ
        # ==================================================

        current_stage = (
            "web_buzz"
        )

        buzz_started = (
            perf_counter()
        )

        try:

            buzz_result = (
                get_web_buzz(
                    question
                )
            )

            web_buzz_ms = (
                perf_counter()
                - buzz_started
            ) * 1000

            buzz_sources = []

            for source in buzz_result.get(
                "sources",
                [],
            ):

                buzz_sources.append(
                    WebBuzzSourceResponse(
                        title=(
                            source.get(
                                "title"
                            )
                            or "Web source"
                        ),

                        url=(
                            source.get(
                                "url"
                            )
                            or ""
                        ),

                        domain=(
                            source.get(
                                "domain"
                            )
                            or ""
                        ),

                        source_type=(
                            source.get(
                                "source_type"
                            )
                            or "WEB"
                        ),

                        published_at=(
                            source.get(
                                "published_at"
                            )
                        ),
                    )
                )

            web_buzz_response = (
                WebBuzzResponse(
                    available=True,

                    summary=(
                        buzz_result.get(
                            "summary"
                        )
                        or ""
                    ),

                    source_class=(
                        buzz_result.get(
                            "source_class"
                        )
                        or "WEB_BUZZ"
                    ),

                    latency_ms=round(
                        web_buzz_ms,
                        2,
                    ),

                    sources=(
                        buzz_sources
                    ),
                )
            )

            log_event(
                "web_buzz_complete",

                request_id=(
                    request_id
                ),

                source_count=len(
                    buzz_sources
                ),

                latency_ms=round(
                    web_buzz_ms,
                    2,
                ),
            )

        except Exception as exc:

            web_buzz_ms = (
                perf_counter()
                - buzz_started
            ) * 1000

            log_event(
                "web_buzz_failed",

                request_id=(
                    request_id
                ),

                error_type=(
                    type(exc).__name__
                ),

                latency_ms=round(
                    web_buzz_ms,
                    2,
                ),
            )

            web_buzz_response = (
                empty_web_buzz_response(
                    web_buzz_ms
                )
            )

        # ==================================================
        # STEP 6 — SOCIAL INTELLIGENCE
        # ==================================================

        current_stage = (
            "social"
        )

        social_started = (
            perf_counter()
        )

        social_insights = []

        # --------------------------------------------------
        # HOT RELOAD CHECK
        # --------------------------------------------------

        reload_status = (
            anlf_index.reload_if_changed()
        )

        if (
            reload_status
            == "reloaded"
        ):

            log_event(
                "social_index_reloaded",

                request_id=(
                    request_id
                ),

                account=(
                    "america_nri_la_frustration"
                ),

                chunk_count=(
                    anlf_index.chunk_count()
                ),
            )

        elif (
            reload_status
            == "reload_failed"
        ):

            log_event(
                "social_index_reload_failed",

                request_id=(
                    request_id
                ),

                account=(
                    "america_nri_la_frustration"
                ),
            )

        elif (
            reload_status
            == "unavailable"
        ):

            log_event(
                "social_index_files_unavailable",

                request_id=(
                    request_id
                ),

                account=(
                    "america_nri_la_frustration"
                ),
            )

        (
            current_anlf_chunks,
            current_anlf_embeddings,
        ) = (
            anlf_index.get_snapshot()
        )

        # --------------------------------------------------
        # ANLF RETRIEVAL + INSIGHT
        # --------------------------------------------------

        try:

            if (
                current_anlf_chunks
                is None
                or current_anlf_embeddings
                is None
            ):

                social_insights.append(
                    empty_anlf_response(
                        caution=(
                            "America NRI Frustration "
                            "index is currently unavailable."
                        )
                    )
                )

            else:

                social_results = (
                    retrieve_anlf(
                        question=(
                            question
                        ),

                        chunks=(
                            current_anlf_chunks
                        ),

                        embeddings=(
                            current_anlf_embeddings
                        ),

                        embedding_model=(
                            embedding_model
                        ),
                    )
                )

                insight = (
                    generate_anlf_insight(
                        question,
                        social_results,
                    )
                )

                social_ms = (
                    perf_counter()
                    - social_started
                ) * 1000

                social_sources = []

                for source in insight.get(
                    "sources",
                    [],
                ):

                    social_sources.append(
                        SocialSourceResponse(
                            account=(
                                source.get(
                                    "account"
                                )
                                or (
                                    "america_nri_"
                                    "la_frustration"
                                )
                            ),

                            platform=(
                                source.get(
                                    "platform"
                                )
                                or "instagram"
                            ),

                            topic=(
                                source.get(
                                    "topic"
                                )
                            ),

                            post_url=(
                                source.get(
                                    "post_url"
                                )
                                or ""
                            ),

                            posted_at=(
                                source.get(
                                    "posted_at"
                                )
                            ),

                            source_class=(
                                source.get(
                                    "source_class"
                                )
                                or (
                                    "SOCIAL_DISCUSSION"
                                )
                            ),
                        )
                    )

                social_insights.append(
                    SocialInsightResponse(
                        account=(
                            "america_nri_la_frustration"
                        ),

                        label=(
                            "America NRI Frustration"
                        ),

                        available=(
                            insight.get(
                                "available",
                                False,
                            )
                        ),

                        summary=(
                            insight.get(
                                "summary",
                                "",
                            )
                        ),

                        admin_points=(
                            insight.get(
                                "admin_points",
                                [],
                            )
                        ),

                        community_points=(
                            insight.get(
                                "community_points",
                                [],
                            )
                        ),

                        caution=(
                            insight.get(
                                "caution",
                                ""
                            )
                        ),

                        source_class=(
                            insight.get(
                                "source_class",
                                "SOCIAL_DISCUSSION",
                            )
                        ),

                        latency_ms=round(
                            social_ms,
                            2,
                        ),

                        sources=(
                            social_sources
                        ),
                    )
                )

                log_event(
                    "social_insight_complete",

                    request_id=(
                        request_id
                    ),

                    account=(
                        "america_nri_la_frustration"
                    ),

                    result_count=len(
                        social_results
                    ),

                    available=(
                        insight.get(
                            "available",
                            False,
                        )
                    ),

                    latency_ms=round(
                        social_ms,
                        2,
                    ),
                )

        except Exception as exc:

            social_ms = (
                perf_counter()
                - social_started
            ) * 1000

            log_event(
                "social_insight_failed",

                request_id=(
                    request_id
                ),

                account=(
                    "america_nri_la_frustration"
                ),

                error_type=(
                    type(exc).__name__
                ),

                latency_ms=round(
                    social_ms,
                    2,
                ),
            )

            social_insights = [
                empty_anlf_response(
                    latency_ms=(
                        social_ms
                    ),

                    caution=(
                        "America NRI Frustration "
                        "community insight is currently "
                        "unavailable."
                    ),
                )
            ]

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        total_ms = (
            perf_counter()
            - total_started
        ) * 1000

        log_event(
            "request_complete",

            request_id=(
                request_id
            ),

            category=(
                category
            ),

            routed_to_rag=True,

            abstained=(
                abstained
            ),

            source_count=len(
                source_rows
            ),

            web_buzz_available=(
                web_buzz_response.available
            ),

            social_available=any(
                item.available
                for item
                in social_insights
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

            web_buzz_ms=round(
                web_buzz_ms,
                2,
            ),

            social_ms=round(
                social_ms,
                2,
            ),

            total_ms=round(
                total_ms,
                2,
            ),
        )

        return AskResponse(
            request_id=(
                request_id
            ),

            question=(
                question
            ),

            category=(
                category
            ),

            routed_to_rag=True,

            answer=(
                generation[
                    "answer"
                ]
            ),

            abstained=(
                abstained
            ),

            latency_ms=round(
                total_ms,
                2,
            ),

            timings_ms=(
                TimingResponse(
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

                    web_buzz_ms=round(
                        web_buzz_ms,
                        2,
                    ),

                    social_ms=round(
                        social_ms,
                        2,
                    ),

                    total_ms=round(
                        total_ms,
                        2,
                    ),
                )
            ),

            sources=(
                source_rows
            ),

            web_buzz=(
                web_buzz_response
            ),

            social_insights=(
                social_insights
            ),
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

            request_id=(
                request_id
            ),

            stage=(
                current_stage
            ),

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