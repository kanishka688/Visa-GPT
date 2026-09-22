import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

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

ACCOUNT = "america_nri_la_frustration"

SOURCE_CLASS = "SOCIAL_DISCUSSION"

ALLOWED_DOMAINS = [
    "imginn.com",
]

RAW_DIR = Path(
    "data/social/anlf/raw"
)

MASTER_FILE = Path(
    "data/social/anlf/anlf_master.jsonl"
)

MAX_POSTS = 10

MAX_COMMENTS_PER_POST = 25


# ==================================================
# HELPERS
# ==================================================


def utc_now_iso() -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def clean_url(
    url: str,
) -> str:

    if not url:

        return ""

    try:

        parsed = urlparse(
            url
        )

        return urlunparse(
            parsed._replace(
                query="",
                fragment="",
            )
        )

    except Exception:

        return url


def make_id(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()[:20]


# ==================================================
# RESPONSE SCHEMA
# ==================================================

ANLF_SCHEMA = {
    "type": "object",

    "properties": {

        "account": {
            "type": "string",
        },

        "posts": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "post_url": {
                        "type": "string",
                    },

                    "posted_at": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },

                    "caption": {
                        "type": "string",
                    },

                    "topic": {
                        "type": "string",
                    },

                    "content_type": {
                        "type": "string",

                        "enum": [
                            "INTERVIEW_EXPERIENCE",
                            "TRAVEL_EXPERIENCE",
                            "STAMPING_EXPERIENCE",
                            "EMPLOYMENT_EXPERIENCE",
                            "ADMIN_GUIDANCE",
                            "GENERAL_COMMUNITY",
                            "OTHER",
                        ],
                    },

                    "admin_info": {
                        "type": "object",

                        "properties": {

                            "summary": {
                                "type": "string",
                            },

                            "interview_questions": {
                                "type": "array",

                                "items": {
                                    "type": "string",
                                },
                            },

                            "experience_notes": {
                                "type": "array",

                                "items": {
                                    "type": "string",
                                },
                            },

                            "travel_notes": {
                                "type": "array",

                                "items": {
                                    "type": "string",
                                },
                            },

                            "stamping_notes": {
                                "type": "array",

                                "items": {
                                    "type": "string",
                                },
                            },

                            "employment_notes": {
                                "type": "array",

                                "items": {
                                    "type": "string",
                                },
                            },

                            "admin_tips": {
                                "type": "array",

                                "items": {
                                    "type": "string",
                                },
                            },
                        },

                        "required": [
                            "summary",
                            "interview_questions",
                            "experience_notes",
                            "travel_notes",
                            "stamping_notes",
                            "employment_notes",
                            "admin_tips",
                        ],

                        "additionalProperties": False,
                    },

                    "comments_count_reported": {
                        "type": [
                            "integer",
                            "null",
                        ],
                    },

                    "comments": {
                        "type": "array",

                        "items": {
                            "type": "object",

                            "properties": {

                                "text": {
                                    "type": "string",
                                },

                                "posted_at": {
                                    "type": [
                                        "string",
                                        "null",
                                    ],
                                },
                            },

                            "required": [
                                "text",
                                "posted_at",
                            ],

                            "additionalProperties": False,
                        },
                    },
                },

                "required": [
                    "post_url",
                    "posted_at",
                    "caption",
                    "topic",
                    "content_type",
                    "admin_info",
                    "comments_count_reported",
                    "comments",
                ],

                "additionalProperties": False,
            },
        },
    },

    "required": [
        "account",
        "posts",
    ],

    "additionalProperties": False,
}


# ==================================================
# WEB SOURCE EXTRACTION
# ==================================================


def extract_web_sources(
    response,
) -> list[str]:

    data = response.model_dump()

    urls = []

    seen = set()

    def walk(
        node: Any,
    ):

        if isinstance(
            node,
            dict,
        ):

            raw_url = node.get(
                "url"
            )

            if (
                isinstance(
                    raw_url,
                    str,
                )
                and raw_url.startswith(
                    (
                        "http://",
                        "https://",
                    )
                )
            ):

                url = clean_url(
                    raw_url
                )

                if (
                    url
                    and url not in seen
                ):

                    seen.add(
                        url
                    )

                    urls.append(
                        url
                    )

            for value in node.values():

                walk(
                    value
                )

        elif isinstance(
            node,
            list,
        ):

            for item in node:

                walk(
                    item
                )

    walk(
        data
    )

    return urls


# ==================================================
# COLLECTION
# ==================================================


def collect_anlf_snapshot() -> dict:

    collected_at = (
        utc_now_iso()
    )

    instructions = f"""
You are collecting public community information for
KK-GPT from the Instagram account:

@{ACCOUNT}

This is a SOCIAL_DISCUSSION dataset.

Search ONLY the allowed public web source.

Find at most {MAX_POSTS} recent public posts
corresponding to @{ACCOUNT}.

The primary goal is to preserve useful INFORMATION,
not the image itself.

Useful information can include:

- visa interview experiences
- visa interview questions
- H-1B interview experiences
- F-1 interview experiences
- stamping experiences
- consular experiences
- travel experiences
- port-of-entry experiences
- immigration employment experiences
- OPT experiences
- STEM OPT experiences
- H-1B experiences
- PERM or I-140 experiences
- layoff experiences
- admin observations
- admin tips
- community experiences shared by the page


==================================================
ADMIN INFORMATION RULE
==================================================

admin_info must contain ONLY information available
from the post caption or other clearly visible
admin/post text.

DO NOT put comment information into admin_info.

A post may describe another person's experience.
That is still acceptable as admin-curated information,
but phrase it as a reported experience.

Do not transform experiences into immigration law.

Do not strengthen statements.

Do not infer facts that were not actually present.


==================================================
CONTENT TYPE
==================================================

Choose exactly one primary content_type:

INTERVIEW_EXPERIENCE
- interview questions or visa interview experience

TRAVEL_EXPERIENCE
- entry, travel, airport, port-of-entry experience

STAMPING_EXPERIENCE
- visa stamping, consulate, Dropbox, appointment,
  passport or related experience

EMPLOYMENT_EXPERIENCE
- layoff, job search, employer, sponsorship,
  OPT/STEM OPT work experience

ADMIN_GUIDANCE
- admin tips, observations, informational guidance

GENERAL_COMMUNITY
- general NRI/community information

OTHER
- does not fit the above


==================================================
ADMIN INFO
==================================================

summary:
Create a concise summary of the useful information
contained in the post caption.

If there is no substantive informational content,
return an empty string.

interview_questions:
Extract actual interview questions mentioned in the
post.

Do not invent likely interview questions.

experience_notes:
Extract concrete experience details reported in the
post.

travel_notes:
Extract concrete travel, entry, airport or
port-of-entry experience details.

stamping_notes:
Extract concrete visa stamping, consular,
appointment, Dropbox or passport-related experience.

employment_notes:
Extract concrete employment, layoff, sponsorship,
job-search, OPT or H-1B employment experiences.

admin_tips:
Extract practical tips explicitly stated by the
post/admin.

If a category is not present, return [].


==================================================
POST DATA
==================================================

For every post:

1. Extract its public post URL.
2. Extract the visible caption.
3. Extract date/time when available.
4. Assign a short immigration/community topic.
5. Assign content_type.
6. Build admin_info.
7. Extract reported comment count if visible.
8. Extract up to {MAX_COMMENTS_PER_POST} comments
   visibly available on the page.


==================================================
COMMENT RULES
==================================================

- Never invent comments.
- Never infer missing comments.
- Do not include commenter usernames.
- Preserve the visible meaning of comment text.
- Comments remain separate from admin_info.
- If no comments are visible, return [].


==================================================
GENERAL RULES
==================================================

- Never invent captions.
- Never invent dates.
- Never invent experiences.
- Never invent interview questions.
- Never infer image-only information that is not
  exposed in the indexed source text.
- posted_at may be null.
- comments_count_reported may be null.
- Do not convert community information into
  official immigration rules.
"""

    user_prompt = f"""
Find recent public posts from @{ACCOUNT}.

Extract:

- post caption
- useful admin information
- interview questions
- reported experiences
- travel experiences
- stamping experiences
- employment experiences
- admin tips
- visible comments

Return only the structured dataset.
"""

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

                tools=[
                    {
                        "type": "web_search",

                        "filters": {
                            "allowed_domains": (
                                ALLOWED_DOMAINS
                            ),
                        },
                    }
                ],

                tool_choice="required",

                include=[
                    (
                        "web_search_call"
                        ".action.sources"
                    ),
                ],

                text={
                    "format": {
                        "type": "json_schema",

                        "name": (
                            "anlf_snapshot"
                        ),

                        "schema": (
                            ANLF_SCHEMA
                        ),

                        "strict": True,
                    }
                },

                max_output_tokens=6000,

                store=False,
            )
        )

    response = (
        execute_openai_request(
            make_request
        )
    )

    payload = json.loads(
        response.output_text
    )

    normalized_posts = []

    for post in payload.get(
        "posts",
        [],
    ):

        post_url = clean_url(
            post.get(
                "post_url",
                "",
            )
        )

        caption = (
            post.get(
                "caption",
                ""
            )
            .strip()
        )

        if (
            not post_url
            or not caption
        ):

            continue

        post_id = make_id(
            post_url
        )

        admin_info = (
            post.get(
                "admin_info"
            )
            or {}
        )

        normalized_admin_info = {
            "summary": (
                admin_info.get(
                    "summary",
                    ""
                )
                .strip()
            ),

            "interview_questions": [
                item.strip()
                for item in admin_info.get(
                    "interview_questions",
                    [],
                )
                if item.strip()
            ],

            "experience_notes": [
                item.strip()
                for item in admin_info.get(
                    "experience_notes",
                    [],
                )
                if item.strip()
            ],

            "travel_notes": [
                item.strip()
                for item in admin_info.get(
                    "travel_notes",
                    [],
                )
                if item.strip()
            ],

            "stamping_notes": [
                item.strip()
                for item in admin_info.get(
                    "stamping_notes",
                    [],
                )
                if item.strip()
            ],

            "employment_notes": [
                item.strip()
                for item in admin_info.get(
                    "employment_notes",
                    [],
                )
                if item.strip()
            ],

            "admin_tips": [
                item.strip()
                for item in admin_info.get(
                    "admin_tips",
                    [],
                )
                if item.strip()
            ],
        }

        comments = []

        seen_comment_ids = set()

        for comment in post.get(
            "comments",
            [],
        ):

            text = (
                comment.get(
                    "text",
                    ""
                )
                .strip()
            )

            if not text:

                continue

            comment_id = make_id(
                f"{post_url}|{text}"
            )

            if (
                comment_id
                in seen_comment_ids
            ):

                continue

            seen_comment_ids.add(
                comment_id
            )

            comments.append(
                {
                    "comment_id": (
                        comment_id
                    ),

                    "text": text,

                    "posted_at": (
                        comment.get(
                            "posted_at"
                        )
                    ),

                    "source_class": (
                        SOURCE_CLASS
                    ),
                }
            )

        normalized_posts.append(
            {
                "post_id": (
                    post_id
                ),

                "platform": (
                    "instagram"
                ),

                "account": (
                    ACCOUNT
                ),

                "post_url": (
                    post_url
                ),

                "posted_at": (
                    post.get(
                        "posted_at"
                    )
                ),

                "caption": (
                    caption
                ),

                "topic": (
                    post.get(
                        "topic",
                        "GENERAL",
                    )
                    .strip()
                ),

                "content_type": (
                    post.get(
                        "content_type",
                        "OTHER",
                    )
                ),

                "admin_info": (
                    normalized_admin_info
                ),

                "comments_count_reported": (
                    post.get(
                        "comments_count_reported"
                    )
                ),

                "comments": (
                    comments
                ),

                "source_class": (
                    SOURCE_CLASS
                ),

                "collected_at": (
                    collected_at
                ),
            }
        )

    return {
        "account": (
            ACCOUNT
        ),

        "platform": (
            "instagram"
        ),

        "source_class": (
            SOURCE_CLASS
        ),

        "collection_method": (
            "PUBLIC_WEB_INDEX"
        ),

        "collected_at": (
            collected_at
        ),

        "posts": (
            normalized_posts
        ),

        "web_sources": (
            extract_web_sources(
                response
            )
        ),
    }


# ==================================================
# RAW SNAPSHOT
# ==================================================


def save_raw_snapshot(
    snapshot: dict,
) -> Path:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    output_path = (
        RAW_DIR
        / f"anlf_{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


# ==================================================
# MASTER DATASET
# ==================================================


def load_master() -> dict:

    posts = {}

    if not MASTER_FILE.exists():

        return posts

    with MASTER_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = (
                line.strip()
            )

            if not line:

                continue

            row = json.loads(
                line
            )

            post_id = row.get(
                "post_id"
            )

            if post_id:

                posts[
                    post_id
                ] = row

    return posts


def merge_snapshot_into_master(
    snapshot: dict,
) -> tuple[int, int]:

    existing_posts = (
        load_master()
    )

    new_posts = 0
    new_comments = 0

    for incoming in snapshot.get(
        "posts",
        [],
    ):

        post_id = incoming[
            "post_id"
        ]

        existing = (
            existing_posts.get(
                post_id
            )
        )

        if existing is None:

            existing_posts[
                post_id
            ] = incoming

            new_posts += 1

            new_comments += len(
                incoming.get(
                    "comments",
                    [],
                )
            )

            continue

        # ------------------------------------------
        # Refresh post/admin information
        # ------------------------------------------

        existing[
            "caption"
        ] = incoming.get(
            "caption",
            existing.get(
                "caption",
                "",
            ),
        )

        existing[
            "topic"
        ] = incoming.get(
            "topic",
            existing.get(
                "topic",
                "GENERAL",
            ),
        )

        existing[
            "content_type"
        ] = incoming.get(
            "content_type",
            existing.get(
                "content_type",
                "OTHER",
            ),
        )

        existing[
            "admin_info"
        ] = incoming.get(
            "admin_info",
            existing.get(
                "admin_info",
                {},
            ),
        )

        if incoming.get(
            "posted_at"
        ):

            existing[
                "posted_at"
            ] = incoming[
                "posted_at"
            ]

        existing[
            "comments_count_reported"
        ] = incoming.get(
            "comments_count_reported"
        )

        existing[
            "collected_at"
        ] = incoming.get(
            "collected_at"
        )

        # ------------------------------------------
        # Merge newly discovered comments
        # ------------------------------------------

        existing_comment_ids = {
            comment.get(
                "comment_id"
            )
            for comment
            in existing.get(
                "comments",
                [],
            )
        }

        for comment in incoming.get(
            "comments",
            [],
        ):

            comment_id = (
                comment.get(
                    "comment_id"
                )
            )

            if (
                comment_id
                and comment_id
                not in existing_comment_ids
            ):

                existing.setdefault(
                    "comments",
                    [],
                ).append(
                    comment
                )

                existing_comment_ids.add(
                    comment_id
                )

                new_comments += 1

    MASTER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MASTER_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in existing_posts.values():

            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )

    return (
        new_posts,
        new_comments,
    )


# ==================================================
# CLI
# ==================================================


def main():

    print(
        "Collecting ANLF public community data..."
    )

    snapshot = (
        collect_anlf_snapshot()
    )

    raw_path = (
        save_raw_snapshot(
            snapshot
        )
    )

    (
        new_posts,
        new_comments,
    ) = merge_snapshot_into_master(
        snapshot
    )

    total_posts = len(
        snapshot.get(
            "posts",
            [],
        )
    )

    total_comments = sum(
        len(
            post.get(
                "comments",
                [],
            )
        )
        for post in snapshot.get(
            "posts",
            [],
        )
    )

    admin_info_posts = sum(
        1
        for post in snapshot.get(
            "posts",
            [],
        )
        if (
            post.get(
                "admin_info",
                {},
            ).get(
                "summary"
            )
        )
    )

    interview_questions = sum(
        len(
            post.get(
                "admin_info",
                {},
            ).get(
                "interview_questions",
                [],
            )
        )
        for post in snapshot.get(
            "posts",
            [],
        )
    )

    print()

    print(
        "=" * 60
    )

    print(
        "ANLF COLLECTION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Posts found:          {total_posts}"
    )

    print(
        f"Comments found:       {total_comments}"
    )

    print(
        f"Admin-info posts:     {admin_info_posts}"
    )

    print(
        f"Interview questions:  {interview_questions}"
    )

    print(
        f"New posts:            {new_posts}"
    )

    print(
        f"New comments:         {new_comments}"
    )

    print(
        f"Web sources:          "
        f"{len(snapshot.get('web_sources', []))}"
    )

    print(
        f"Raw snapshot:         {raw_path}"
    )

    print(
        f"Master dataset:       {MASTER_FILE}"
    )


if __name__ == "__main__":

    main()