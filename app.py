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
    "for F-1, CPT, OPT, STEM OPT, H-1B, H-4, "
    "PERM and I-140."
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
        "Example: What is CPT?"
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
        "Checking official sources..."
    ):

        try:

            response = requests.post(
                ASK_URL,
                json={
                    "question": (
                        cleaned_question
                    )
                },
                timeout=120,
            )

            response.raise_for_status()

            result = response.json()

        except requests.RequestException as exc:

            st.error(
                "KK-GPT could not process "
                "the request."
            )

            st.stop()


    # ==================================================
    # ANSWER
    # ==================================================

    st.subheader(
        "Answer"
    )

    st.markdown(
        result.get(
            "answer",
            "No answer returned.",
        )
    )


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

    if abstained:

        st.caption(
            "KK-GPT abstained rather than "
            "answer without sufficient evidence."
        )


    # ==================================================
    # SOURCES
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
                    f"Open source [{citation_id}]",
                    url,
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

                "timings_ms": result.get(
                    "timings_ms",
                    {},
                ),
            }
        )