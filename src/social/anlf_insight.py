import json

from src.core.llm_client import (
    call_llm_json,
)

from src.social.anlf_retrieval import (
    load_index,
    retrieve_anlf,
)


# ==================================================
# CONFIG
# ==================================================

MAX_EVIDENCE_CHUNKS = 5


# ==================================================
# STRUCTURED OUTPUT
# ==================================================

ANLF_INSIGHT_SCHEMA = {
    "type": "object",

    "properties": {

        "summary": {
            "type": "string",
        },

        "admin_points": {
            "type": "array",

            "items": {
                "type": "string",
            },
        },

        "community_points": {
            "type": "array",

            "items": {
                "type": "string",
            },
        },

        "caution": {
            "type": "string",
        },
    },

    "required": [
        "summary",
        "admin_points",
        "community_points",
        "caution",
    ],

    "additionalProperties": False,
}


# ==================================================
# EVIDENCE
# ==================================================


def build_anlf_evidence(
    results: list[dict],
) -> str:

    blocks = []

    for index, result in enumerate(
        results[
            :MAX_EVIDENCE_CHUNKS
        ],
        start=1,
    ):

        chunk_type = result.get(
            "chunk_type",
            "UNKNOWN",
        )

        topic = result.get(
            "topic",
            "",
        )

        raw_text = result.get(
            "raw_text",
            "",
        )

        url = result.get(
            "post_url",
            "",
        )

        score = result.get(
            "score",
            0.0,
        )

        blocks.append(
            "\n".join(
                [
                    f"[ANLF {index}]",
                    f"Type: {chunk_type}",
                    f"Topic: {topic}",
                    f"Similarity: {score:.4f}",
                    f"Source: {url}",
                    "",
                    raw_text,
                ]
            )
        )

    return "\n\n".join(
        blocks
    )


# ==================================================
# SOURCES
# ==================================================


def build_sources(
    results: list[dict],
) -> list[dict]:

    sources = []

    seen_urls = set()

    for result in results:

        url = result.get(
            "post_url"
        )

        if (
            not url
            or url in seen_urls
        ):

            continue

        seen_urls.add(
            url
        )

        sources.append(
            {
                "account": (
                    result.get(
                        "account"
                    )
                    or "america_nri_la_frustration"
                ),

                "platform": (
                    result.get(
                        "platform"
                    )
                    or "instagram"
                ),

                "topic": (
                    result.get(
                        "topic"
                    )
                ),

                "post_url": (
                    url
                ),

                "posted_at": (
                    result.get(
                        "posted_at"
                    )
                ),

                "source_class": (
                    "SOCIAL_DISCUSSION"
                ),
            }
        )

    return sources


# ==================================================
# GENERATION
# ==================================================


def generate_anlf_insight(
    question: str,
    results: list[dict],
) -> dict:

    if not results:

        return {
            "available": False,

            "summary": "",

            "admin_points": [],

            "community_points": [],

            "caution": (
                "No sufficiently relevant "
                "America NRI Frustration content "
                "was found for this question."
            ),

            "sources": [],

            "source_class": (
                "SOCIAL_DISCUSSION"
            ),
        }

    evidence = (
        build_anlf_evidence(
            results
        )
    )

    system_prompt = """
You are the America NRI Frustration community
insight layer for KK-GPT.

You are NOT an official immigration source.

You are given retrieved content collected from the
public America NRI Frustration community page.

The retrieved evidence may contain:

- admin/post information
- admin-curated experiences
- interview questions
- travel experiences
- stamping experiences
- employment experiences
- community comments


==================================================
CRITICAL GROUNDING RULES
==================================================

Use ONLY the supplied ANLF evidence.

Never add immigration rules from your own knowledge.

Never treat the page or its commenters as an
authoritative legal source.

Never convert an experience into a general rule.

Never strengthen a claim beyond what the evidence
actually says.

Never invent interview questions.

Never invent experiences.

Never invent community sentiment.

If only admin/post evidence exists, community_points
must be [].

If only comments exist, admin_points may be [].

Ignore retrieved evidence that is not relevant to
the user's question even if it was supplied.


==================================================
SUMMARY
==================================================

summary should briefly explain what relevant
information this page currently contains about the
user's question.

Use language such as:

- "The page reports..."
- "A recent post discusses..."
- "The page highlights..."
- "Community comments include..."

Do NOT say:

- "The law says..."
- "You must..."
- "USCIS requires..."

unless those exact words are merely being described
as something the page itself claimed, and make that
attribution explicit.


==================================================
ADMIN POINTS
==================================================

admin_points are useful facts, claims, experiences,
questions, warnings, or tips contained in ADMIN_INFO
evidence.

Each point must remain attributed to the page/post.

Prefer concise useful information.


==================================================
COMMUNITY POINTS
==================================================

community_points must come ONLY from COMMENT
evidence.

Summarize recurring or useful community experiences.

Do not imply a single comment represents community
consensus.

If comment evidence is sparse, describe it narrowly.

Do not include usernames.


==================================================
CAUTION
==================================================

Return a concise caution explaining that this is
community/admin-reported information and not
official immigration guidance.
"""

    user_prompt = f"""
User question:

{question}

America NRI Frustration evidence:

{evidence}

Create a concise, evidence-grounded community
insight.
"""

    insight = call_llm_json(
        system_prompt=(
            system_prompt
        ),

        user_prompt=(
            user_prompt
        ),

        schema_name=(
            "anlf_insight"
        ),

        schema=(
            ANLF_INSIGHT_SCHEMA
        ),

        max_output_tokens=1200,
    )

    return {
        "available": True,

        "summary": (
            insight.get(
                "summary",
                "",
            )
        ),

        "admin_points": (
            insight.get(
                "admin_points",
                [],
            )
        ),

        "community_points": (
            insight.get(
                "community_points",
                [],
            )
        ),

        "caution": (
            insight.get(
                "caution",
                (
                    "This is community-reported "
                    "information, not official "
                    "immigration guidance."
                ),
            )
        ),

        "sources": (
            build_sources(
                results
            )
        ),

        "source_class": (
            "SOCIAL_DISCUSSION"
        ),
    }


# ==================================================
# CLI
# ==================================================


def main():

    print(
        "Loading ANLF social index..."
    )

    (
        chunks,
        embeddings,
        embedding_model,
    ) = load_index()

    question = input(
        "ANLF question: "
    ).strip()

    results = retrieve_anlf(
        question=question,

        chunks=chunks,

        embeddings=embeddings,

        embedding_model=(
            embedding_model
        ),
    )

    insight = (
        generate_anlf_insight(
            question,
            results,
        )
    )

    print()

    print(
        "=" * 60
    )

    print(
        "AMERICA NRI FRUSTRATION"
    )

    print(
        "=" * 60
    )

    if not insight[
        "available"
    ]:

        print()

        print(
            insight[
                "caution"
            ]
        )

        return

    print()

    print(
        insight[
            "summary"
        ]
    )

    if insight[
        "admin_points"
    ]:

        print()

        print(
            "ADMIN / POST INSIGHT"
        )

        for point in insight[
            "admin_points"
        ]:

            print(
                f"- {point}"
            )

    if insight[
        "community_points"
    ]:

        print()

        print(
            "COMMUNITY COMMENTS"
        )

        for point in insight[
            "community_points"
        ]:

            print(
                f"- {point}"
            )

    print()

    print(
        "CAUTION"
    )

    print(
        insight[
            "caution"
        ]
    )

    if insight[
        "sources"
    ]:

        print()

        print(
            "SOURCES"
        )

        for index, source in enumerate(
            insight[
                "sources"
            ],
            start=1,
        ):

            print(
                f"[{index}] "
                f"{source.get('topic')}"
            )

            print(
                source[
                    "post_url"
                ]
            )


if __name__ == "__main__":

    main()