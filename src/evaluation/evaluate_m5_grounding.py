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
    build_evidence,
    evidence_to_text,
)

from src.core.llm_client import (
    call_llm_json,
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

CITATION_REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/m5_citation_metrics.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/m5_grounding_metrics.json"
)


# ==================================================
# CONFIG
# ==================================================

RETRIEVAL_TOP_K = 5


# ==================================================
# HELPERS
# ==================================================


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def safe_int(
    value,
):

    try:
        return max(
            0,
            int(value),
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0


# ==================================================
# LOAD GENERATED ANSWERS
# ==================================================


def load_generated_answers():

    report = json.loads(
        CITATION_REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    rows = report.get(
        "results",
        [],
    )

    # Only evaluate answers that actually made it
    # through the evidence gate and generation and
    # had valid citation IDs.
    return [
        row
        for row in rows
        if (
            row.get("status")
            == "PASS"
            and row.get("answer")
        )
    ]


# ==================================================
# GROUNDING JUDGE
# ==================================================


def judge_grounding(
    question,
    answer,
    evidence_items,
):

    evidence_text = (
        evidence_to_text(
            evidence_items
        )
    )

    system_prompt = """
You are the grounding evaluator for KK-GPT,
a U.S. immigration RAG system.

Your task is ONLY to evaluate whether a generated
answer is grounded in the supplied official
evidence.

Do NOT answer the immigration question yourself.

Do NOT use outside knowledge.

Evaluate the generated answer exactly as written.

Definitions:

MATERIAL CLAIM
A factual statement that contributes to answering
the user's immigration question.

Examples:
- eligibility rules
- deadlines
- durations
- government procedures
- who authorizes something
- legal consequences
- form purposes
- employment requirements

Do not count generic statements such as:
"This is informational guidance, not legal advice."

SUPPORTED CLAIM
A material factual claim that is directly supported
by at least one supplied evidence block.

CITED CLAIM
A material factual claim followed by one or more
citation markers such as [1], [2], or [1][3].

SEMANTICALLY CORRECT CITATION
The cited evidence block actually supports the
claim immediately associated with that citation.

IMPORTANT:

A citation number merely existing is NOT enough.

For example:

Claim:
"The H-1B grace period may be up to 60 days. [2]"

If SOURCE [2] discusses only prevailing wages,
the citation is semantically incorrect.

Another example:

Claim:
"Not every H-4 spouse is automatically authorized
to work. [1]"

If SOURCE [1] states that only certain H-4 spouses
qualify under specified conditions, that claim is
supported.

Negative answers are valid when the evidence
supports the negative conclusion.

Do not require exact wording.
Reasonable paraphrases are acceptable.

A claim is unsupported if it introduces a factual
detail that cannot be derived from the supplied
evidence.

Set:

answers_question=true
only if the answer substantially addresses the
user's exact question.

all_material_claims_cited=true
only if every material factual claim has an
appropriate inline citation.

all_citations_support_claims=true
only if every citation used for a material claim
actually supports that claim.

no_unsupported_material_claims=true
only if every material factual claim is supported
by the supplied evidence.

Count:

material_claims
supported_material_claims
cited_material_claims
correctly_supported_citation_claims

For unsupported_claims:
return a list containing each unsupported material
claim. Return an empty list when none exist.

For citation_mismatches:
return a list describing each citation that does
not support its associated claim. Return an empty
list when none exist.

Return only the structured result required by
the provided schema.
"""

    user_prompt = f"""
QUESTION:

{question}


GENERATED ANSWER:

{answer}


OFFICIAL EVIDENCE:

{evidence_text}
"""

    grounding_schema = {
        "type": "object",
        "properties": {
            "answers_question": {
                "type": "boolean",
            },
            "all_material_claims_cited": {
                "type": "boolean",
            },
            "all_citations_support_claims": {
                "type": "boolean",
            },
            "no_unsupported_material_claims": {
                "type": "boolean",
            },
            "material_claims": {
                "type": "integer",
                "minimum": 0,
            },
            "supported_material_claims": {
                "type": "integer",
                "minimum": 0,
            },
            "cited_material_claims": {
                "type": "integer",
                "minimum": 0,
            },
            "correctly_supported_citation_claims": {
                "type": "integer",
                "minimum": 0,
            },
            "unsupported_claims": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "citation_mismatches": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "answers_question",
            "all_material_claims_cited",
            "all_citations_support_claims",
            "no_unsupported_material_claims",
            "material_claims",
            "supported_material_claims",
            "cited_material_claims",
            "correctly_supported_citation_claims",
            "unsupported_claims",
            "citation_mismatches",
            "reason",
        ],
        "additionalProperties": False,
    }

    try:

        parsed = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="grounding_evaluation",
            schema=grounding_schema,
            temperature=0.0,
            max_output_tokens=2000,
        )

    except Exception as exc:

        return {
            "judge_valid": False,

            "answers_question": False,

            "all_material_claims_cited": False,

            "all_citations_support_claims": False,

            "no_unsupported_material_claims": False,

            "fully_grounded": False,

            "material_claims": 0,

            "supported_material_claims": 0,

            "cited_material_claims": 0,

            "correctly_supported_citation_claims": 0,

            "unsupported_claims": [],

            "citation_mismatches": [],

            "reason": (
                "Grounding judge provider failure: "
                f"{type(exc).__name__}"
            ),
        }

    answers_question = (
        parsed.get(
            "answers_question"
        )
        is True
    )

    all_material_claims_cited = (
        parsed.get(
            "all_material_claims_cited"
        )
        is True
    )

    all_citations_support_claims = (
        parsed.get(
            "all_citations_support_claims"
        )
        is True
    )

    no_unsupported_material_claims = (
        parsed.get(
            "no_unsupported_material_claims"
        )
        is True
    )

    # Derive the aggregate ourselves instead of
    # trusting a model-produced fully_grounded field.
    fully_grounded = (
        answers_question
        and all_material_claims_cited
        and all_citations_support_claims
        and no_unsupported_material_claims
    )

    material_claims = safe_int(
        parsed.get(
            "material_claims"
        )
    )

    supported_material_claims = min(
        material_claims,
        safe_int(
            parsed.get(
                "supported_material_claims"
            )
        ),
    )

    cited_material_claims = min(
        material_claims,
        safe_int(
            parsed.get(
                "cited_material_claims"
            )
        ),
    )

    correctly_supported_citation_claims = min(
        cited_material_claims,
        safe_int(
            parsed.get(
                "correctly_supported_citation_claims"
            )
        ),
    )

    unsupported_claims = parsed.get(
        "unsupported_claims",
        [],
    )

    citation_mismatches = parsed.get(
        "citation_mismatches",
        [],
    )

    return {
        "judge_valid": True,

        "answers_question": (
            answers_question
        ),

        "all_material_claims_cited": (
            all_material_claims_cited
        ),

        "all_citations_support_claims": (
            all_citations_support_claims
        ),

        "no_unsupported_material_claims": (
            no_unsupported_material_claims
        ),

        "fully_grounded": (
            fully_grounded
        ),

        "material_claims": (
            material_claims
        ),

        "supported_material_claims": (
            supported_material_claims
        ),

        "cited_material_claims": (
            cited_material_claims
        ),

        "correctly_supported_citation_claims": (
            correctly_supported_citation_claims
        ),

        "unsupported_claims": (
            unsupported_claims
        ),

        "citation_mismatches": (
            citation_mismatches
        ),

        "reason": (
            parsed.get(
                "reason",
                "",
            )
        ),
    }


# ==================================================
# EVALUATION
# ==================================================


def evaluate():

    answers = (
        load_generated_answers()
    )

    print(
        "=" * 60
    )

    print(
        "M5 SEMANTIC GROUNDING EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Generated answers: "
        f"{len(answers)}"
    )

    # ==================================================
    # LOAD RETRIEVAL
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

    valid_judgments = 0

    answers_question_count = 0

    fully_grounded_count = 0

    all_claims_cited_count = 0

    semantic_citation_pass_count = 0

    no_unsupported_claims_count = 0

    total_material_claims = 0

    total_supported_claims = 0

    total_cited_claims = 0

    total_correct_citation_claims = 0

    failures = []

    results_report = []

    # ==================================================
    # RUN
    # ==================================================

    for index, item in enumerate(
        answers,
        start=1,
    ):

        question_id = item[
            "id"
        ]

        question = item[
            "question"
        ]

        answer = item[
            "answer"
        ]

        # Reconstruct the same deterministic
        # top-5 evidence used during generation.

        query_info = (
            understand_query(
                question
            )
        )

        retrieval_results = (
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

                top_k=RETRIEVAL_TOP_K,
            )
        )

        evidence_items = (
            build_evidence(
                retrieval_results
            )
        )

        judgment = (
            judge_grounding(
                question=question,
                answer=answer,
                evidence_items=(
                    evidence_items
                ),
            )
        )

        if judgment[
            "judge_valid"
        ]:

            valid_judgments += 1

        if judgment[
            "answers_question"
        ]:

            answers_question_count += 1

        if judgment[
            "fully_grounded"
        ]:

            fully_grounded_count += 1

        if judgment[
            "all_material_claims_cited"
        ]:

            all_claims_cited_count += 1

        if judgment[
            "all_citations_support_claims"
        ]:

            semantic_citation_pass_count += 1

        if judgment[
            "no_unsupported_material_claims"
        ]:

            no_unsupported_claims_count += 1

        total_material_claims += (
            judgment[
                "material_claims"
            ]
        )

        total_supported_claims += (
            judgment[
                "supported_material_claims"
            ]
        )

        total_cited_claims += (
            judgment[
                "cited_material_claims"
            ]
        )

        total_correct_citation_claims += (
            judgment[
                "correctly_supported_citation_claims"
            ]
        )

        row = {
            "id": (
                question_id
            ),

            "question": (
                question
            ),

            "answer": (
                answer
            ),

            **judgment,

            "evidence_source_ids": [
                evidence.get(
                    "source_id"
                )
                for evidence
                in evidence_items
            ],
        }

        results_report.append(
            row
        )

        if not judgment[
            "fully_grounded"
        ]:

            failures.append(
                row
            )

        status = (
            "PASS"
            if judgment[
                "fully_grounded"
            ]
            else "FAIL"
        )

        print(
            f"[{index}/{len(answers)}] "
            f"{question_id}: {status}"
        )

    # ==================================================
    # METRICS
    # ==================================================

    total_answers = len(
        answers
    )

    answer_relevance_rate = (
        safe_divide(
            answers_question_count,
            total_answers,
        )
    )

    fully_grounded_rate = (
        safe_divide(
            fully_grounded_count,
            total_answers,
        )
    )

    claim_support_rate = (
        safe_divide(
            total_supported_claims,
            total_material_claims,
        )
    )

    claim_citation_coverage = (
        safe_divide(
            total_cited_claims,
            total_material_claims,
        )
    )

    semantic_citation_precision = (
        safe_divide(
            total_correct_citation_claims,
            total_cited_claims,
        )
    )

    unsupported_claim_free_rate = (
        safe_divide(
            no_unsupported_claims_count,
            total_answers,
        )
    )

    hallucination_answer_rate = (
        1.0
        - unsupported_claim_free_rate
    )

    # ==================================================
    # REPORT
    # ==================================================

    report = {

        "dataset": {
            "generated_answers": (
                total_answers
            ),

            "valid_judgments": (
                valid_judgments
            ),
        },

        "metrics": {

            "answer_relevance_rate": (
                answer_relevance_rate
            ),

            "fully_grounded_answer_rate": (
                fully_grounded_rate
            ),

            "claim_support_rate": (
                claim_support_rate
            ),

            "claim_citation_coverage": (
                claim_citation_coverage
            ),

            "semantic_citation_precision": (
                semantic_citation_precision
            ),

            "unsupported_claim_free_rate": (
                unsupported_claim_free_rate
            ),

            "hallucination_answer_rate": (
                hallucination_answer_rate
            ),

            "answers_question_count": (
                answers_question_count
            ),

            "fully_grounded_count": (
                fully_grounded_count
            ),

            "all_claims_cited_count": (
                all_claims_cited_count
            ),

            "semantic_citation_pass_count": (
                semantic_citation_pass_count
            ),

            "no_unsupported_claims_count": (
                no_unsupported_claims_count
            ),

            "total_material_claims": (
                total_material_claims
            ),

            "supported_material_claims": (
                total_supported_claims
            ),

            "cited_material_claims": (
                total_cited_claims
            ),

            "correctly_supported_citation_claims": (
                total_correct_citation_claims
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
    # FINAL OUTPUT
    # ==================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "M5 GROUNDING RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Generated answers:             "
        f"{total_answers}"
    )

    print(
        f"Valid judge outputs:           "
        f"{valid_judgments}"
    )

    print()

    print(
        f"Answer relevance:              "
        f"{answer_relevance_rate * 100:.1f}%"
    )

    print(
        f"Fully grounded answers:        "
        f"{fully_grounded_rate * 100:.1f}%"
    )

    print(
        f"Claim support rate:            "
        f"{claim_support_rate * 100:.1f}%"
    )

    print(
        f"Claim citation coverage:       "
        f"{claim_citation_coverage * 100:.1f}%"
    )

    print(
        f"Semantic citation precision:   "
        f"{semantic_citation_precision * 100:.1f}%"
    )

    print(
        f"Unsupported-claim-free rate:   "
        f"{unsupported_claim_free_rate * 100:.1f}%"
    )

    print(
        f"Hallucination answer rate:     "
        f"{hallucination_answer_rate * 100:.1f}%"
    )

    print(
        f"\nGrounding failures: "
        f"{len(failures)}"
    )

    if failures:

        print(
            "\nFAILED ANSWERS"
        )

        for item in failures:

            print(
                f"\n- {item['id']}"
            )

            print(
                "  Judge valid: "
                f"{item['judge_valid']}"
            )

            print(
                "  Answers question: "
                f"{item['answers_question']}"
            )

            print(
                "  All claims cited: "
                f"{item['all_material_claims_cited']}"
            )

            print(
                "  Citations support claims: "
                f"{item['all_citations_support_claims']}"
            )

            print(
                "  No unsupported claims: "
                f"{item['no_unsupported_material_claims']}"
            )

            print(
                "  Reason: "
                f"{item['reason']}"
            )

            if item[
                "unsupported_claims"
            ]:

                print(
                    "  Unsupported claims:"
                )

                for claim in (
                    item[
                        "unsupported_claims"
                    ]
                ):

                    print(
                        f"    - {claim}"
                    )

            if item[
                "citation_mismatches"
            ]:

                print(
                    "  Citation mismatches:"
                )

                for mismatch in (
                    item[
                        "citation_mismatches"
                    ]
                ):

                    print(
                        f"    - {mismatch}"
                    )

    print(
        "\nNOTE:"
    )

    print(
        "This is an LLM-assisted grounding "
        "evaluation, so it is a diagnostic metric, "
        "not absolute human gold."
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