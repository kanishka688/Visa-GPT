from typing import Any
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

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

MAX_BUZZ_SOURCES = 8


# Government / official sources belong in the
# OFFICIAL RAG channel, not the Web Buzz channel.

BLOCKED_OFFICIAL_DOMAINS = [
    "uscis.gov",
    "dhs.gov",
    "studyinthestates.dhs.gov",
    "ice.gov",
    "cbp.gov",
    "ecfr.gov",
    "dol.gov",
    "foreignlaborcert.doleta.gov",
    "state.gov",
    "travel.state.gov",
    "govinfo.gov",
    "federalregister.gov",
]


TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
}


# ==================================================
# URL HELPERS
# ==================================================


def clean_url(
    url: str,
) -> str:
    """
    Remove common tracking parameters while
    preserving the actual source URL.
    """

    try:

        parsed = urlparse(
            url
        )

        clean_query = [
            (
                key,
                value,
            )
            for key, value
            in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
            if key.lower()
            not in TRACKING_QUERY_KEYS
        ]

        return urlunparse(
            parsed._replace(
                query=urlencode(
                    clean_query
                ),
                fragment="",
            )
        )

    except Exception:

        return url


def get_domain(
    url: str,
) -> str:

    try:

        domain = (
            urlparse(
                url
            )
            .netloc
            .lower()
        )

        if domain.startswith(
            "www."
        ):
            domain = domain[4:]

        return domain

    except Exception:

        return ""


def is_official_domain(
    url: str,
) -> bool:
    """
    Web Buzz should not contain government
    sources because those belong to official RAG.
    """

    domain = get_domain(
        url
    )

    if not domain:

        return False

    if domain.endswith(
        ".gov"
    ):
        return True

    for blocked_domain in (
        BLOCKED_OFFICIAL_DOMAINS
    ):

        if (
            domain
            == blocked_domain
            or domain.endswith(
                f".{blocked_domain}"
            )
        ):
            return True

    return False


# ==================================================
# SOURCE CLASSIFICATION
# ==================================================


def classify_source(
    url: str,
) -> str:

    domain = get_domain(
        url
    )

    if domain.endswith(
        "reddit.com"
    ):
        return "COMMUNITY"

    if domain in {
        "quora.com",
    }:
        return "COMMUNITY"

    if domain in {
        "news.google.com",
    }:
        return "NEWS"

    return "WEB"


def build_source_title(
    title: str | None,
    url: str,
) -> str:
    """
    Prefer the source-provided title.

    If OpenAI returns no useful title, create a
    readable label from the URL.
    """

    if (
        title
        and title.strip()
        and title.strip().lower()
        != "web source"
    ):

        return title.strip()

    domain = get_domain(
        url
    )

    parsed = urlparse(
        url
    )

    # ----------------------------------------------
    # REDDIT
    # ----------------------------------------------

    if domain.endswith(
        "reddit.com"
    ):

        parts = [
            part
            for part in parsed.path.split(
                "/"
            )
            if part
        ]

        if (
            len(parts) >= 2
            and parts[0] == "r"
        ):

            return (
                f"Reddit — r/{parts[1]}"
            )

        return "Reddit"

    # ----------------------------------------------
    # GENERIC DOMAIN
    # ----------------------------------------------

    if domain:

        readable_domain = (
            domain
            .replace(
                ".com",
                "",
            )
            .replace(
                ".org",
                "",
            )
            .replace(
                ".net",
                "",
            )
            .replace(
                "-",
                " ",
            )
        )

        return readable_domain.title()

    return "Web source"


# ==================================================
# SOURCE EXTRACTION
# ==================================================


def extract_web_sources(
    response,
    max_sources=MAX_BUZZ_SOURCES,
):
    """
    Extract and normalize web-search sources.

    Government sources are excluded because they
    belong in the official RAG channel.
    """

    data = response.model_dump()

    sources = []

    seen_urls = set()

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
                    url not in seen_urls
                    and not is_official_domain(
                        url
                    )
                ):

                    seen_urls.add(
                        url
                    )

                    raw_title = (
                        node.get(
                            "title"
                        )
                        or node.get(
                            "name"
                        )
                    )

                    sources.append(
                        {
                            "title": (
                                build_source_title(
                                    raw_title,
                                    url,
                                )
                            ),
                            "url": url,
                            "domain": (
                                get_domain(
                                    url
                                )
                            ),
                            "source_type": (
                                classify_source(
                                    url
                                )
                            ),
                            "published_at": (
                                node.get(
                                    "published_date"
                                )
                                or node.get(
                                    "published_at"
                                )
                                or node.get(
                                    "date"
                                )
                            ),
                        }
                    )

            for value in (
                node.values()
            ):

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

    return sources[
        :max_sources
    ]


# ==================================================
# WEB BUZZ SEARCH
# ==================================================


def get_web_buzz(
    question: str,
) -> dict:
    """
    Search current web discussion related to a
    U.S. immigration question.

    IMPORTANT:

    This is NOT official legal evidence.

    Government information belongs exclusively
    in the official RAG channel.
    """

    cleaned_question = (
        question.strip()
    )

    if not cleaned_question:

        return {
            "summary": "",
            "sources": [],
            "source_class": (
                "WEB_BUZZ"
            ),
        }

    instructions = """
You are the Web Buzz analyst for KK-GPT,
a U.S. immigration information system.

This section represents CURRENT PUBLIC DISCUSSION.

It is NOT the official immigration answer.

Search for recent web-accessible discussion from:

- community forums
- Reddit
- immigration communities
- immigration blogs
- news reporting
- commentary
- worker/student discussions
- other public web conversation

DO NOT use government websites as sources for this
section.

Official government information is handled by a
separate KK-GPT RAG pipeline.

IMPORTANT RULES:

1. Summarize what people are currently discussing,
   reporting, experiencing, asking, or worrying
   about.

2. Prefer recent discussion.

3. Look for recurring themes rather than isolated
   comments.

4. Clearly describe anecdotal experiences as
   anecdotes.

5. Clearly describe claims or commentary as claims
   or commentary.

6. Never convert community experience into an
   immigration rule.

7. Do not give personalized legal advice.

8. Do not predict an individual's immigration case.

9. Avoid presenting rumors as established facts.

10. If sources disagree, describe the disagreement.

11. Do not use USCIS, DHS, DOL, eCFR, State
    Department, Federal Register, GovInfo, or other
    government sources in this Web Buzz section.

12. Do not provide a legal-rule explanation merely
    because the topic involves immigration.

13. Focus on what is receiving attention NOW.

14. Return approximately 3 to 5 concise bullet
    points when meaningful current discussion exists.

15. If meaningful current discussion is sparse,
    say so clearly.

16. Do not create a separate Sources section.
    KK-GPT renders source links separately.

Begin with one concise sentence describing the
overall discussion pattern.

Then provide the key recurring themes as bullets.
"""

    user_prompt = f"""
Search the current public web for recent community,
news, forum, blog, and commentary discussion related
to this U.S. immigration topic:

{cleaned_question}

Focus on what people are currently discussing,
experiencing, reporting, asking, or worrying about.

Do NOT use government sources.

Do NOT answer the official immigration-law question.
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

                input=user_prompt,

                reasoning={
                    "effort": "low",
                },

                tools=[
                    {
                        "type": (
                            "web_search"
                        ),

                        "filters": {
                            "blocked_domains": (
                                BLOCKED_OFFICIAL_DOMAINS
                            ),
                        },
                    }
                ],

                tool_choice=(
                    "required"
                ),

                include=[
                    (
                        "web_search_call"
                        ".action.sources"
                    ),
                ],

                max_output_tokens=900,

                store=False,
            )
        )

    response = (
        execute_openai_request(
            make_request
        )
    )

    summary = (
        response.output_text
        or ""
    ).strip()

    sources = extract_web_sources(
        response
    )

    return {
        "summary": summary,
        "sources": sources,
        "source_class": "WEB_BUZZ",
    }


# ==================================================
# COMMAND-LINE TEST
# ==================================================


def main():

    question = input(
        "Buzz question: "
    ).strip()

    result = get_web_buzz(
        question
    )

    print()

    print(
        "=" * 60
    )

    print(
        "CURRENT WEB BUZZ"
    )

    print(
        "=" * 60
    )

    print()

    print(
        result[
            "summary"
        ]
    )

    print()

    print(
        "=" * 60
    )

    print(
        "WEB BUZZ SOURCES"
    )

    print(
        "=" * 60
    )

    if not result[
        "sources"
    ]:

        print(
            "\nNo non-government "
            "buzz sources returned."
        )

        return

    for index, source in enumerate(
        result[
            "sources"
        ],
        start=1,
    ):

        print()

        print(
            f"[{index}] "
            f"{source['title']}"
        )

        print(
            "Type: "
            f"{source['source_type']}"
        )

        print(
            "Domain: "
            f"{source['domain']}"
        )

        print(
            source[
                "url"
            ]
        )

        if source.get(
            "published_at"
        ):

            print(
                "Published: "
                f"{source['published_at']}"
            )


if __name__ == "__main__":

    main()