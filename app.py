import os

import requests
import streamlit as st


# ==================================================
# CONFIG
# ==================================================

API_BASE_URL = os.getenv(
    "KKGPT_API_BASE_URL",
    "http://127.0.0.1:8000",
)

ASK_URL = f"{API_BASE_URL}/ask"
HEALTH_URL = f"{API_BASE_URL}/health"


# ==================================================
# PAGE
# ==================================================

st.set_page_config(
    page_title="KK-GPT",
    page_icon="⚖️",
    layout="centered",
)


# ==================================================
# HEADER
# ==================================================

st.title("KK-GPT")

st.caption(
    "Official-source U.S. immigration information "
    "with current community and web discussion."
)

st.info(
    "KK-GPT provides informational guidance from "
    "official sources and is not legal advice."
)


# ==================================================
# API HEALTH
# ==================================================


def check_api_health():

    try:

        response = requests.get(
            HEALTH_URL,
            timeout=5,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:

        return None


health = check_api_health()

if health is None:

    st.error(
        "KK-GPT backend is unavailable. "
        "Start the FastAPI service first."
    )

    st.stop()


if health.get("status") != "ok":

    st.warning(
        "KK-GPT backend is starting up."
    )

    st.stop()


# ==================================================
# QUESTION
# ==================================================

question = st.text_area(
    "Ask an immigration question",
    placeholder=(
        "Example: What happens to an "
        "H-1B worker after a layoff?"
    ),
    height=100,
)


ask_clicked = st.button(
    "Ask KK-GPT",
    type="primary",
    use_container_width=True,
)


# ==================================================
# REQUEST
# ==================================================

if ask_clicked:

    cleaned_question = (
        question.strip()
    )

    if not cleaned_question:

        st.warning(
            "Enter a question first."
        )

        st.stop()

    with st.spinner(
        "Checking official sources "
        "and current web discussion..."
    ):

        try:

            response = requests.post(
                ASK_URL,
                json={
                    "question": (
                        cleaned_question
                    )
                },
                timeout=180,
            )

            response.raise_for_status()

            result = response.json()

        except requests.RequestException:

            st.error(
                "KK-GPT could not process "
                "the request."
            )

            st.stop()


    # ==================================================
    # ROUTING
    # ==================================================

    category = result.get(
        "category",
        "UNKNOWN",
    )

    routed_to_rag = result.get(
        "routed_to_rag",
        False,
    )

    abstained = result.get(
        "abstained",
        True,
    )


    # ==================================================
    # OFFICIAL ANSWER
    # ==================================================

    st.subheader(
        "Official Answer"
    )

    st.markdown(
        result.get(
            "answer",
            "No answer returned.",
        )
    )

    if abstained:

        st.caption(
            "KK-GPT abstained rather than "
            "answer without sufficient "
            "official evidence."
        )


    # ==================================================
    # OFFICIAL SOURCES
    # ==================================================

    sources = result.get(
        "sources",
        [],
    )

    if sources:

        st.subheader(
            "Official Sources"
        )

        for source in sources:

            citation_id = source.get(
                "citation_id"
            )

            agency = source.get(
                "agency"
            ) or "Official source"

            document = source.get(
                "document"
            ) or "Document"

            url = source.get(
                "url"
            )

            authority_type = source.get(
                "authority_type"
            )

            st.markdown(
                f"**[{citation_id}] "
                f"{agency} — {document}**"
            )

            if authority_type:

                st.caption(
                    f"Authority: "
                    f"{authority_type}"
                )

            if url:

                st.link_button(
                    f"Open official source "
                    f"[{citation_id}]",
                    url,
                )


    # ==================================================
    # DIVIDER
    # ==================================================

    st.divider()


    # ==================================================
    # CURRENT WEB BUZZ
    # ==================================================

    web_buzz = result.get(
        "web_buzz",
        {},
    )

    buzz_available = web_buzz.get(
        "available",
        False,
    )

    st.subheader(
        web_buzz.get(
            "label",
            "Current Web Buzz",
        )
    )

    st.caption(
        web_buzz.get(
            "disclaimer",
            (
                "Community, news, forum, and web "
                "discussion only. This is not "
                "official immigration evidence."
            ),
        )
    )

    if buzz_available:

        buzz_summary = web_buzz.get(
            "summary",
            "",
        )

        if buzz_summary:

            st.markdown(
                buzz_summary
            )

        else:

            st.caption(
                "No meaningful current discussion "
                "was returned."
            )


        # ==================================================
        # WEB BUZZ SOURCES
        # ==================================================

        buzz_sources = web_buzz.get(
            "sources",
            [],
        )

        if buzz_sources:

            with st.expander(
                "Web Buzz Sources",
                expanded=False,
            ):

                for index, source in enumerate(
                    buzz_sources,
                    start=1,
                ):

                    title = source.get(
                        "title"
                    ) or "Web source"

                    url = source.get(
                        "url"
                    )

                    domain = source.get(
                        "domain"
                    )

                    source_type = source.get(
                        "source_type"
                    ) or "WEB"

                    published_at = source.get(
                        "published_at"
                    )

                    st.markdown(
                        f"**[{index}] {title}**"
                    )

                    source_details = [
                        source_type
                    ]

                    if domain:

                        source_details.append(
                            domain
                        )

                    if published_at:

                        source_details.append(
                            str(
                                published_at
                            )
                        )

                    st.caption(
                        " · ".join(
                            source_details
                        )
                    )

                    if url:

                        st.link_button(
                            f"Open buzz source "
                            f"[{index}]",
                            url,
                        )

                    if index < len(
                        buzz_sources
                    ):

                        st.markdown(
                            "---"
                        )

    else:

        st.caption(
            "Current Web Buzz is unavailable "
            "for this request. The official "
            "answer above is unaffected."
        )


    # ==================================================
    # DEBUG / METRICS
    # ==================================================

    with st.expander(
        "Request details"
    ):

        st.write(
            {
                "request_id": result.get(
                    "request_id"
                ),

                "category": category,

                "routed_to_rag": (
                    routed_to_rag
                ),

                "abstained": (
                    abstained
                ),

                "web_buzz_available": (
                    buzz_available
                ),

                "timings_ms": result.get(
                    "timings_ms",
                    {},
                ),
            }
        )