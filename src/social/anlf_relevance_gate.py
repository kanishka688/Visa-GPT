import json
from pathlib import Path

from src.core.llm_client import (
    OPENAI_CLIENT,
    execute_openai_request,
)

from src.core.settings import (
    OPENAI_MODEL,
)


# ==================================================
# CONFIG
# ==================================================

MASTER_FILE = Path(
    "data/social/anlf/anlf_master.jsonl"
)

RELEVANT_FILE = Path(
    "data/social/anlf/anlf_relevant.jsonl"
)

REPORT_FILE = Path(
    "data/social/anlf/anlf_relevance_report.json"
)

BATCH_SIZE = 20


# ==================================================
# RELEVANCE CATEGORIES
# ==================================================

RELEVANCE_CATEGORIES = [
    "VISA_IMMIGRATION",
    "STAMPING_CONSULAR",
    "IMMIGRATION_TRAVEL",
    "PARENTS_RELATIVES_VISITING",
    "STAY_WORK_RULES",
    "IMMIGRATION_POLICY_POLITICS",
    "GENERAL_IMMIGRATION",
]


# ==================================================
# STRUCTURED OUTPUT
# ==================================================

RELEVANCE_SCHEMA = {
    "type": "object",

    "properties": {

        "results": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "post_id": {
                        "type": "string",
                    },

                    "index_eligible": {
                        "type": "boolean",
                    },

                    "categories": {
                        "type": "array",

                        "items": {
                            "type": "string",

                            "enum": (
                                RELEVANCE_CATEGORIES
                            ),
                        },
                    },

                    "reason": {
                        "type": "string",
                    },
                },

                "required": [
                    "post_id",
                    "index_eligible",
                    "categories",
                    "reason",
                ],

                "additionalProperties": False,
            },
        },
    },

    "required": [
        "results",
    ],

    "additionalProperties": False,
}


# ==================================================
# LOAD DATA
# ==================================================


def load_master_posts() -> list[dict]:

    if not MASTER_FILE.exists():

        raise FileNotFoundError(
            f"Master dataset not found: "
            f"{MASTER_FILE}"
        )

    posts = []

    with MASTER_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:

                continue

            posts.append(
                json.loads(
                    line
                )
            )

    return posts


# ==================================================
# CLASSIFICATION INPUT
# ==================================================


def build_classification_post(
    post: dict,
) -> dict:

    admin_info = (
        post.get(
            "admin_info"
        )
        or {}
    )

    comments = post.get(
        "comments",
        [],
    )

    return {
        "post_id": (
            post.get(
                "post_id",
                "",
            )
        ),

        "topic": (
            post.get(
                "topic",
                "",
            )
        ),

        "content_type": (
            post.get(
                "content_type",
                "",
            )
        ),

        "caption": (
            post.get(
                "caption",
                "",
            )
        ),

        "admin_summary": (
            admin_info.get(
                "summary",
                "",
            )
        ),

        "interview_questions": (
            admin_info.get(
                "interview_questions",
                [],
            )
        ),

        "experience_notes": (
            admin_info.get(
                "experience_notes",
                [],
            )
        ),

        "travel_notes": (
            admin_info.get(
                "travel_notes",
                [],
            )
        ),

        "stamping_notes": (
            admin_info.get(
                "stamping_notes",
                [],
            )
        ),

        "employment_notes": (
            admin_info.get(
                "employment_notes",
                [],
            )
        ),

        "admin_tips": (
            admin_info.get(
                "admin_tips",
                [],
            )
        ),

        # Comments can provide context, but should
        # not make an otherwise unrelated post
        # relevant based on a casual mention.
        "comments": [
            comment.get(
                "text",
                "",
            )
            for comment
            in comments[:10]
            if comment.get(
                "text"
            )
        ],
    }


# ==================================================
# RELEVANCE CLASSIFIER
# ==================================================


def classify_batch(
    posts: list[dict],
) -> list[dict]:

    instructions = """
You are the relevance gate for the KK-GPT
America NRI Frustration community dataset.

Your only task is to decide whether each post is
useful for a U.S. visa / immigration information
product.

==================================================
KEEP THE POST
==================================================

Set index_eligible=true if the post is meaningfully
related to ANY of the following:


1. VISA / IMMIGRATION

Examples include, but are not limited to:

- F-1
- F-2
- CPT
- OPT
- STEM OPT
- H-1B
- H-4
- H-4 EAD
- B-1
- B-2
- L-1
- L-2
- O-1
- J-1
- J-2
- PERM
- PWD
- I-140
- I-485
- adjustment of status
- green card
- employment immigration
- family immigration
- dependent visas
- immigration status
- visa validity
- extensions
- change of status
- work authorization


2. VISA STAMPING / CONSULAR PROCESSING

Keep content involving:

- visa stamping
- visa interviews
- interview questions
- consulates
- embassies
- Dropbox
- interview waiver
- appointment availability
- appointment scheduling
- appointment delays
- passport submission
- passport return
- administrative processing
- 221(g)
- stamping experiences
- consular experiences


3. IMMIGRATION-RELATED U.S. TRAVEL

Keep content involving:

- entering the United States
- re-entering the United States
- port of entry
- CBP
- immigration inspection
- airport immigration questions
- immigration-document checks
- travel while a petition is pending
- travel while status is pending
- travel after visa stamping
- travel with F-1 / H-1B / H-4 / other status
- advance parole
- visa-related international travel

Travel must have a meaningful immigration,
visa, entry, or status connection.

A normal road trip, airline story, robotaxi story,
vacation story, or transportation story is NOT
enough.


4. PARENTS / RELATIVES VISITING THE U.S.

Keep content involving:

- parents visiting the United States
- relatives visiting the United States
- B-1/B-2 visitor visas
- visitor visa interviews
- visitor visa stamping
- invitation letters
- supporting documents
- parents' interview experiences
- relatives' interview experiences
- visitor port-of-entry experiences
- CBP questions for visitors
- visitor length of stay
- I-94 issues
- visitor extensions
- visitor status problems


5. RULES AFFECTING STAYING OR WORKING IN THE U.S.

Keep content involving:

- grace periods
- layoffs affecting immigration status
- status maintenance
- unlawful presence
- work authorization
- employment authorization
- sponsorship
- employer changes
- portability
- visa expiration
- immigration deadlines
- status extensions
- immigration consequences of job loss


6. IMMIGRATION POLICY / POLITICS

Keep political content when immigration is
materially involved.

Examples:

- immigration executive orders
- immigration legislation
- proposed immigration rules
- DHS immigration proposals
- USCIS policy changes
- State Department visa policy
- immigration-related court decisions
- political proposals affecting immigrants
- political proposals affecting visa holders
- immigration enforcement changes
- green-card policy changes
- student-visa policy
- employment-visa policy

Do NOT reject something merely because it is
political.

If the political development materially concerns
immigration, visas, immigrants, visa holders,
visitors, immigration status, entry, employment
authorization, or ability to remain in the U.S.,
KEEP IT.


7. GENERAL IMMIGRATION

Keep any other post that is meaningfully about:

- immigrants in the United States
- visa holders
- international students
- employment-based immigration
- family immigration
- immigration processing
- immigration experiences
- immigration backlogs
- immigration procedures
- immigration-related life in the United States


==================================================
DO NOT KEEP
==================================================

Set index_eligible=false when the post is clearly
unrelated to the above.

Examples:

- robotaxis
- cars
- normal domestic travel
- shopping
- restaurants
- entertainment
- celebrity news
- unrelated crime
- unrelated local news
- general finance
- stock market news
- general technology news
- general politics with no meaningful immigration
  connection
- elections with no meaningful immigration issue
- unrelated tax or dividend proposals
- general U.S. news unrelated to immigrants,
  visas, visitors, status, entry, or immigration


==================================================
IMPORTANT
==================================================

Use the actual subject of the post.

Do not keep an unrelated post merely because the
account itself is an NRI account.

A casual immigration-related comment under an
otherwise unrelated post is not enough by itself.

If the post is substantially relevant to immigration
or visa-related life in the United States, prefer
keeping it.

Assign every relevant category that materially
applies.

For an irrelevant post, return categories=[].

Do not evaluate whether the post's claims are true.

This gate determines relevance only.
"""

    payload = [
        build_classification_post(
            post
        )
        for post in posts
    ]

    user_prompt = (
        "Classify these ANLF posts for the "
        "KK-GPT social immigration index:\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    def make_request():

        return (
            OPENAI_CLIENT
            .responses
            .create(
                model=OPENAI_MODEL,

                instructions=(
                    instructions
                ),

                input=(
                    user_prompt
                ),

                reasoning={
                    "effort": "low",
                },

                text={
                    "format": {
                        "type": "json_schema",

                        "name": (
                            "anlf_relevance"
                        ),

                        "schema": (
                            RELEVANCE_SCHEMA
                        ),

                        "strict": True,
                    }
                },

                max_output_tokens=3000,

                store=False,
            )
        )

    response = (
        execute_openai_request(
            make_request
        )
    )

    result = json.loads(
        response.output_text
    )

    return result.get(
        "results",
        [],
    )


# ==================================================
# CLASSIFY ALL POSTS
# ==================================================


def classify_all_posts(
    posts: list[dict],
) -> dict[str, dict]:

    classifications = {}

    for start in range(
        0,
        len(posts),
        BATCH_SIZE,
    ):

        batch = posts[
            start:
            start + BATCH_SIZE
        ]

        results = (
            classify_batch(
                batch
            )
        )

        for result in results:

            post_id = result.get(
                "post_id"
            )

            if post_id:

                classifications[
                    post_id
                ] = result

    return classifications


# ==================================================
# WRITE RELEVANT DATASET
# ==================================================


def write_relevant_dataset(
    posts: list[dict],
    classifications: dict[str, dict],
) -> tuple[int, int]:

    RELEVANT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relevant_count = 0
    excluded_count = 0

    with RELEVANT_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for post in posts:

            post_id = post.get(
                "post_id"
            )

            classification = (
                classifications.get(
                    post_id,
                    {
                        "index_eligible": False,
                        "categories": [],
                        "reason": (
                            "No relevance classification returned."
                        ),
                    },
                )
            )

            if not classification.get(
                "index_eligible",
                False,
            ):

                excluded_count += 1
                continue

            enriched_post = {
                **post,

                "relevance": {
                    "index_eligible": True,

                    "categories": (
                        classification.get(
                            "categories",
                            [],
                        )
                    ),

                    "reason": (
                        classification.get(
                            "reason",
                            "",
                        )
                    ),
                },
            }

            handle.write(
                json.dumps(
                    enriched_post,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )

            relevant_count += 1

    return (
        relevant_count,
        excluded_count,
    )


# ==================================================
# REPORT
# ==================================================


def write_report(
    posts: list[dict],
    classifications: dict[str, dict],
):

    rows = []

    for post in posts:

        post_id = post.get(
            "post_id"
        )

        classification = (
            classifications.get(
                post_id,
                {}
            )
        )

        rows.append(
            {
                "post_id": post_id,

                "topic": post.get(
                    "topic"
                ),

                "post_url": post.get(
                    "post_url"
                ),

                "index_eligible": (
                    classification.get(
                        "index_eligible",
                        False,
                    )
                ),

                "categories": (
                    classification.get(
                        "categories",
                        [],
                    )
                ),

                "reason": (
                    classification.get(
                        "reason",
                        "",
                    )
                ),
            }
        )

    REPORT_FILE.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ==================================================
# CLI
# ==================================================


def main():

    print(
        "Running ANLF relevance gate..."
    )

    posts = (
        load_master_posts()
    )

    classifications = (
        classify_all_posts(
            posts
        )
    )

    (
        relevant_count,
        excluded_count,
    ) = write_relevant_dataset(
        posts,
        classifications,
    )

    write_report(
        posts,
        classifications,
    )

    print()

    print(
        "=" * 60
    )

    print(
        "ANLF RELEVANCE GATE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Master posts:       {len(posts)}"
    )

    print(
        f"Relevant posts:     {relevant_count}"
    )

    print(
        f"Excluded posts:     {excluded_count}"
    )

    print(
        f"Index-ready file:   {RELEVANT_FILE}"
    )

    print(
        f"Gate report:        {REPORT_FILE}"
    )

    print()

    print(
        "Relevant:"
    )

    for post in posts:

        result = classifications.get(
            post.get(
                "post_id"
            ),
            {},
        )

        if result.get(
            "index_eligible"
        ):

            print(
                "  KEEP   "
                f"{post.get('topic')}"
            )

    print()

    print(
        "Excluded:"
    )

    for post in posts:

        result = classifications.get(
            post.get(
                "post_id"
            ),
            {},
        )

        if not result.get(
            "index_eligible",
            False,
        ):

            print(
                "  DROP   "
                f"{post.get('topic')}"
            )


if __name__ == "__main__":

    main()