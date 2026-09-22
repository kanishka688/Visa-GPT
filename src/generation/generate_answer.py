import re

from src.core.llm_client import (
    call_llm,
    call_llm_json,
)

from src.core.settings import (
    MAX_EVIDENCE_SOURCES,
)


# ==================================================
# CONFIG
# ==================================================

ABSTAIN_MESSAGE = (
    "I do not have enough official evidence "
    "to answer this question."
)


# ==================================================
# EVIDENCE PREPARATION
# ==================================================


def build_evidence(
    results,
    max_sources=MAX_EVIDENCE_SOURCES,
):
    """
    Convert ranked retrieval results into numbered
    evidence blocks.

    Each citation number maps to an exact retrieved
    chunk rather than to generic model knowledge.
    """

    evidence_items = []

    for number, result in enumerate(
        results[:max_sources],
        start=1,
    ):
        chunk = result.get(
            "chunk",
            {},
        )

        evidence_items.append(
            {
                "citation_id": number,
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
                "topic": chunk.get(
                    "topic"
                ),
                "url": chunk.get(
                    "url"
                ),
                "authority_type": (
                    result.get(
                        "authority_type"
                    )
                    or chunk.get(
                        "authority_type"
                    )
                ),
                "text": chunk.get(
                    "text",
                    "",
                ),
            }
        )

    return evidence_items


def evidence_to_text(
    evidence_items,
):
    """
    Format retrieved evidence for the LLM.
    """

    blocks = []

    for item in evidence_items:

        blocks.append(
            f"""
SOURCE [{item['citation_id']}]

Agency:
{item['agency']}

Document:
{item['document']}

Authority:
{item['authority_type']}

Topic:
{item['topic']}

Evidence:
{item['text']}
"""
        )

    return "\n".join(
        blocks
    )


# ==================================================
# EVIDENCE SUFFICIENCY GATE
# ==================================================


def assess_evidence(
    question,
    results,
):
    """
    Decide whether retrieved official evidence
    actually supports answering the question.

    Query policy:
        Should this TYPE of question go to RAG?

    Evidence gate:
        Did retrieval actually find enough evidence?
    """

    evidence_items = build_evidence(
        results
    )

    if not evidence_items:

        return {
            "supported": False,
            "reason": (
                "No official evidence was retrieved."
            ),
        }

    evidence_text = evidence_to_text(
        evidence_items
    )

    system_prompt = """
You are the evidence-sufficiency checker for
KK-GPT, a U.S. immigration RAG system.

Your ONLY task is to determine whether the
provided official evidence is sufficient to answer
the user's exact question.

Do NOT answer the immigration question.

Do NOT use outside knowledge.

Do NOT assume facts that are not stated in the
provided evidence.

Mark supported=true only when the evidence
directly supports the core answer.

Topic similarity alone is NOT enough.

Examples:

Question:
"What is Form ETA-9089?"

Evidence:
A document discussing Form I-140.

Result:
supported=false

Question:
"Who authorizes CPT?"

Evidence:
Official guidance stating that the DSO authorizes
CPT in SEVIS and on Form I-20.

Result:
supported=true

Question:
"What is the H-1B grace period?"

Evidence:
Official text explaining the post-employment
grace period.

Result:
supported=true

If a material part of the question cannot be
answered from the evidence, use supported=false.

Return only the structured result required by
the provided schema.
"""

    user_prompt = f"""
QUESTION:

{question}

OFFICIAL EVIDENCE:

{evidence_text}
"""

    evidence_schema = {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean",
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "supported",
            "reason",
        ],
        "additionalProperties": False,
    }

    try:

        parsed = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="evidence_sufficiency",
            schema=evidence_schema,
            temperature=0.0,
            max_output_tokens=500,
        )

    except Exception as exc:

        return {
            "supported": False,
            "reason": (
                "Evidence checker provider failure: "
                f"{type(exc).__name__}"
            ),
        }

    supported = (
        parsed.get(
            "supported"
        )
        is True
    )

    return {
        "supported": supported,
        "reason": parsed.get(
            "reason",
            "",
        ),
    }


# ==================================================
# CITATION VALIDATION
# ==================================================


def extract_citation_ids(
    answer,
):
    """
    Extract citation numbers such as:

        [1]
        [2]
        [1][3]
    """

    matches = re.findall(
        r"\[(\d+)\]",
        answer,
    )

    return [
        int(value)
        for value in matches
    ]


def validate_citations(
    answer,
    evidence_items,
):
    """
    Deterministic citation-ID validation.

    Requirements:

    - at least one citation exists
    - every citation maps to supplied evidence
    """

    valid_ids = {
        item[
            "citation_id"
        ]
        for item in evidence_items
    }

    cited_ids = extract_citation_ids(
        answer
    )

    if not cited_ids:

        return {
            "valid": False,
            "reason": (
                "Generated answer contains "
                "no citations."
            ),
            "cited_ids": [],
            "invalid_ids": [],
        }

    invalid_ids = sorted(
        {
            citation_id
            for citation_id in cited_ids
            if citation_id
            not in valid_ids
        }
    )

    if invalid_ids:

        return {
            "valid": False,
            "reason": (
                "Generated answer used citation "
                "numbers that do not exist."
            ),
            "cited_ids": cited_ids,
            "invalid_ids": invalid_ids,
        }

    return {
        "valid": True,
        "reason": "Citations are valid.",
        "cited_ids": cited_ids,
        "invalid_ids": [],
    }


# ==================================================
# GROUNDED GENERATION
# ==================================================


def generate_answer_text(
    question,
    evidence_items,
):
    """
    Generate an answer using only supplied
    official evidence.
    """

    evidence_text = evidence_to_text(
        evidence_items
    )

    system_prompt = f"""
You are KK-GPT, a U.S. immigration information
assistant.

Your job is to answer using ONLY the official
evidence supplied in the prompt.

GROUNDING RULES

1. Answer the user's exact question directly.

2. Do NOT use outside knowledge, even when you
   already know the answer.

3. Do NOT add background facts, definitions,
   organizational descriptions, procedural details,
   legal mechanisms, examples, or exceptions unless
   they are explicitly supported by the supplied
   evidence.

4. Every material factual statement MUST have an
   inline citation immediately after the statement.

5. The opening sentence or direct conclusion is also
   a material factual statement and MUST be cited.

Example:

BAD:
"Yes, CPT may be available immediately in some
graduate programs."

GOOD:
"Yes, CPT may be available immediately in some
graduate programs. [2]"

6. Do NOT make an uncited introductory conclusion
   and then cite supporting details later.

7. A citation may ONLY refer to one of the numbered
   evidence blocks supplied below.

8. Use a citation only when that specific evidence
   block supports the entire claim immediately before
   the citation.

9. If one sentence contains multiple factual claims,
   ensure the cited source supports ALL of them.

   If not, split the sentence into separate claims
   with separate citations.

10. Do NOT attach extra citations merely because the
    sources discuss the same topic.

11. Prefer the narrowest statement directly supported
    by the evidence.

12. Do NOT strengthen evidence.

For example:

If evidence says:
"Students file Form I-765 with USCIS."

Do NOT expand this into:
"USCIS authorizes employment by approving Form I-765
and issuing the EAD"

unless the supplied evidence explicitly supports
those additional facts.

13. Do NOT infer organizational relationships.

For example, do not say an office is a division of
another agency unless the evidence explicitly says
so.

14. Do NOT convert partial evidence into a broader
    categorical statement.

15. Reasonable paraphrasing is allowed, but the
    factual meaning must remain within what the
    evidence directly supports.

16. If multiple sources genuinely support the same
    claim, citations may appear as:

[1][2]

17. Do NOT create citation numbers that were not
    supplied.

18. Prefer concise answers. Fewer fully supported
    claims are better than a detailed answer containing
    unsupported additions.

19. Mention exceptions only when they materially
    affect the answer AND the exception is supported
    by the evidence.

20. Do NOT predict USCIS, DOL, or Department of
    State decisions.

21. If the supplied evidence is insufficient to
    answer the question safely, respond exactly:

"{ABSTAIN_MESSAGE}"

22. Do NOT include a separate Sources section.
    KK-GPT generates sources separately.

23. This is informational guidance, not legal advice.

FINAL SELF-CHECK BEFORE RESPONDING

For every factual sentence ask:

- Is this exact claim supported by supplied evidence?
- Is a citation immediately attached?
- Does that cited source support the entire claim?
- Did I add any detail not present in the evidence?

If any answer is no, remove or rewrite that claim.

Return only the final user-facing answer.
"""

    user_prompt = f"""
QUESTION:

{question}

OFFICIAL EVIDENCE:

{evidence_text}
"""

    return call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_output_tokens=700,
    ).strip()


# ==================================================
# FULL GENERATION PIPELINE
# ==================================================


def generate_grounded_answer(
    question,
    results,
):
    """
    Full RAG generation pipeline:

        ranked evidence
             ↓
        evidence sufficiency
             ↓
        answer / abstain
             ↓
        inline citations
             ↓
        citation validation
    """

    evidence_items = build_evidence(
        results
    )

    # ----------------------------------------------
    # STEP 1 — EVIDENCE GATE
    # ----------------------------------------------

    evidence_check = assess_evidence(
        question,
        results,
    )

    if not evidence_check[
        "supported"
    ]:

        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "evidence_check": (
                evidence_check
            ),
            "citation_check": None,
            "sources": evidence_items,
        }

    # ----------------------------------------------
    # STEP 2 — GENERATION
    # ----------------------------------------------

    try:

        answer = generate_answer_text(
            question,
            evidence_items,
        )

    except Exception as exc:

        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "evidence_check": (
                evidence_check
            ),
            "citation_check": {
                "valid": False,
                "reason": (
                    "Generation provider failure: "
                    f"{type(exc).__name__}"
                ),
                "cited_ids": [],
                "invalid_ids": [],
            },
            "sources": evidence_items,
        }

    # Model can independently decide that the
    # supplied evidence is insufficient.

    if (
        ABSTAIN_MESSAGE.lower()
        in answer.lower()
    ):

        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "evidence_check": (
                evidence_check
            ),
            "citation_check": None,
            "sources": evidence_items,
        }

    # ----------------------------------------------
    # STEP 3 — CITATION VALIDATION
    # ----------------------------------------------

    citation_check = validate_citations(
        answer,
        evidence_items,
    )

    if not citation_check[
        "valid"
    ]:

        return {
            "answer": ABSTAIN_MESSAGE,
            "abstained": True,
            "evidence_check": (
                evidence_check
            ),
            "citation_check": (
                citation_check
            ),
            "sources": evidence_items,
        }

    # ----------------------------------------------
    # SUCCESS
    # ----------------------------------------------

    return {
        "answer": answer,
        "abstained": False,
        "evidence_check": (
            evidence_check
        ),
        "citation_check": (
            citation_check
        ),
        "sources": evidence_items,
    }