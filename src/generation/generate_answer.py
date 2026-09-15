import json
import re

import requests


# ==================================================
# CONFIG
# ==================================================

from src.core.settings import (
    LLM_MODEL,
    OLLAMA_URL,
    MAX_EVIDENCE_SOURCES,
)

MAX_EVIDENCE_SOURCES = 5

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
# OLLAMA HELPER
# ==================================================


def call_ollama(
    messages,
    json_mode=False,
    temperature=0.0,
):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    if json_mode:
        payload[
            "format"
        ] = "json"

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()[
        "message"
    ][
        "content"
    ]


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

    This is different from query policy:

    query policy:
        Should this TYPE of question go to RAG?

    evidence gate:
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
KK-Gpt, a U.S. immigration RAG system.

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

Return JSON only:

{
  "supported": true,
  "reason": "brief debugging explanation"
}
"""

    user_prompt = f"""
QUESTION:

{question}


OFFICIAL EVIDENCE:

{evidence_text}
"""

    raw = call_ollama(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        json_mode=True,
        temperature=0.0,
    )

    try:

        parsed = json.loads(
            raw
        )

    except json.JSONDecodeError:

        # Fail closed.
        return {
            "supported": False,
            "reason": (
                "Evidence checker returned "
                "invalid JSON."
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
    Basic deterministic citation validation.

    Requirements:
    - at least one citation exists
    - every cited number maps to retrieved evidence
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
    evidence_text = evidence_to_text(
        evidence_items
    )

    system_prompt = f"""
You are KK-Gpt, a U.S. immigration information
assistant.

Answer ONLY from the official evidence supplied
below.

Rules:

1. Answer the user's exact question directly.

2. Do not use outside knowledge.

3. Do not invent immigration rules.

4. Every material factual claim must include an
   inline citation such as [1] or [2].

5. A citation number may ONLY refer to the numbered
   official evidence provided to you.

6. Put the citation immediately after the claim it
   supports.

7. If multiple sources support a claim, you may use:
   [1][2]

8. Do not cite a source unless its evidence actually
   supports the claim.

9. Do not create fake citation numbers.

10. Prefer concise answers for simple questions.

11. Mention exceptions only when they materially
    affect the answer.

12. Do not predict USCIS, DOL, or Department of
    State decisions.

13. If the evidence is insufficient, respond exactly:

"{ABSTAIN_MESSAGE}"

14. Do not include a separate Sources section.
    KK-Gpt will generate that separately.

15. This is informational guidance, not legal advice.
"""

    user_prompt = f"""
QUESTION:

{question}


OFFICIAL EVIDENCE:

{evidence_text}
"""

    return call_ollama(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
    ).strip()


# ==================================================
# FULL GENERATION PIPELINE
# ==================================================


def generate_grounded_answer(
    question,
    results,
):
    """
    Full RAG V1 generation pipeline:

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

    answer = generate_answer_text(
        question,
        evidence_items,
    )

    # Model can still independently decide
    # evidence is insufficient.
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

        # Fail closed for V1 rather than returning
        # an uncited or incorrectly cited answer.
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