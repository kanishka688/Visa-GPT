from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET

import fitz
import requests
import trafilatura


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ==================================================
# PATHS
# ==================================================

SOURCES_PATH = (
    PROJECT_ROOT
    / "config/sources.json"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data/raw"
)


# ==================================================
# HTTP CONFIG
# ==================================================

REQUEST_TIMEOUT = 60

HEADERS = {
    "User-Agent": (
        "KK-GPT-RAG/1.0 "
        "(educational immigration RAG project)"
    )
}


# ==================================================
# LOAD SOURCES
# ==================================================


def load_sources():

    return json.loads(
        SOURCES_PATH.read_text(
            encoding="utf-8"
        )
    )


# ==================================================
# TEXT CLEANING
# ==================================================


def clean_text(
    text,
):
    if not text:
        return ""

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ==================================================
# PDF INGESTION
# ==================================================


def extract_pdf_text(
    content,
):
    pages = []

    document = fitz.open(
        stream=content,
        filetype="pdf",
    )

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        page_text = clean_text(
            page.get_text(
                "text"
            )
        )

        pages.append(
            {
                "page": page_number,
                "text": page_text,
            }
        )

    combined_text = "\n\n".join(
        page[
            "text"
        ]
        for page in pages
        if page.get(
            "text"
        )
    )

    return {
        "text": clean_text(
            combined_text
        ),
        "pages": pages,
    }


def ingest_pdf(
    source,
):
    response = requests.get(
        source[
            "url"
        ],
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return extract_pdf_text(
        response.content
    )


# ==================================================
# NORMAL HTML INGESTION
# ==================================================


def ingest_html(
    source,
):
    response = requests.get(
        source[
            "url"
        ],
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    downloaded = response.text

    extracted = trafilatura.extract(
        downloaded,
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_precision=False,
        favor_recall=True,
    )

    if not extracted:
        raise ValueError(
            "HTML extraction returned no text."
        )

    return {
        "text": clean_text(
            extracted
        ),
        "pages": [],
    }


# ==================================================
# eCFR URL PARSING
# ==================================================


def parse_ecfr_url(
    url,
):
    """
    Example:

    https://www.ecfr.gov/current/title-8/
    chapter-I/subchapter-B/part-214/
    subpart-A/section-214.2

    Returns:

    {
        "title": "8",
        "part": "214",
        "subpart": "A",
        "section": "214.2"
    }
    """

    title_match = re.search(
        r"/title-([^/]+)",
        url,
        re.IGNORECASE,
    )

    part_match = re.search(
        r"/part-([^/]+)",
        url,
        re.IGNORECASE,
    )

    subpart_match = re.search(
        r"/subpart-([^/]+)",
        url,
        re.IGNORECASE,
    )

    section_match = re.search(
        r"/section-([^/?#]+)",
        url,
        re.IGNORECASE,
    )

    chapter_match = re.search(
        r"/chapter-([^/]+)",
        url,
        re.IGNORECASE,
    )

    if not title_match:
        raise ValueError(
            f"Could not determine eCFR title "
            f"from URL: {url}"
        )

    return {
        "title": title_match.group(
            1
        ),
        "chapter": (
            chapter_match.group(
                1
            )
            if chapter_match
            else None
        ),
        "part": (
            part_match.group(
                1
            )
            if part_match
            else None
        ),
        "subpart": (
            subpart_match.group(
                1
            )
            if subpart_match
            else None
        ),
        "section": (
            section_match.group(
                1
            )
            if section_match
            else None
        ),
    }


# ==================================================
# eCFR CURRENT DATE
# ==================================================


def get_ecfr_latest_date(
    title_number,
):
    """
    Use the title's current up_to_date_as_of value
    rather than blindly assuming today's date.
    """

    url = (
        "https://www.ecfr.gov"
        "/api/versioner/v1/titles.json"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    titles = payload.get(
        "titles",
        []
    )

    for title in titles:

        if str(
            title.get(
                "number"
            )
        ) == str(
            title_number
        ):

            latest_date = title.get(
                "up_to_date_as_of"
            )

            if latest_date:
                return latest_date

    raise ValueError(
        f"Could not determine latest eCFR date "
        f"for Title {title_number}."
    )


# ==================================================
# eCFR XML → TEXT
# ==================================================


def ecfr_xml_to_text(
    xml_content,
):
    """
    Extract readable regulatory text from eCFR XML.

    ElementTree.itertext() preserves the actual
    regulation content regardless of eCFR XML tag
    names such as DIV, HD, P, FP, etc.
    """

    try:

        root = ET.fromstring(
            xml_content
        )

    except ET.ParseError as exc:

        raise ValueError(
            f"Could not parse eCFR XML: {exc}"
        ) from exc

    text_parts = []

    for value in root.itertext():

        value = clean_text(
            value
        )

        if value:
            text_parts.append(
                value
            )

    text = "\n".join(
        text_parts
    )

    return clean_text(
        text
    )


# ==================================================
# eCFR INGESTION
# ==================================================


def ingest_ecfr(
    source,
):
    """
    Fetch eCFR content from the official Versioner
    API instead of scraping the JavaScript-heavy
    website.

    Endpoint:

    /api/versioner/v1/full/{date}/title-{title}.xml

    We request the smallest hierarchy available:
    section > subpart > part.
    """

    hierarchy = parse_ecfr_url(
        source[
            "url"
        ]
    )

    title = hierarchy[
        "title"
    ]

    latest_date = (
        get_ecfr_latest_date(
            title
        )
    )

    api_url = (
        "https://www.ecfr.gov"
        f"/api/versioner/v1/full/"
        f"{latest_date}/"
        f"title-{title}.xml"
    )

    params = {}

    # Part is useful even for section requests.
    if hierarchy.get(
        "part"
    ):
        params[
            "part"
        ] = hierarchy[
            "part"
        ]

    # Lowest hierarchy wins.
    if hierarchy.get(
        "section"
    ):

        params[
            "section"
        ] = hierarchy[
            "section"
        ]

    elif hierarchy.get(
        "subpart"
    ):

        params[
            "subpart"
        ] = hierarchy[
            "subpart"
        ]

    elif hierarchy.get(
        "chapter"
    ):

        params[
            "chapter"
        ] = hierarchy[
            "chapter"
        ]

    print(
        f"    eCFR API date: "
        f"{latest_date}"
    )

    print(
        f"    eCFR params: "
        f"{params}"
    )

    response = requests.get(
        api_url,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    text = ecfr_xml_to_text(
        response.text
    )

    if len(text) < 100:

        raise ValueError(
            "eCFR API returned suspiciously "
            "little regulatory text."
        )

    return {
        "text": text,
        "pages": [],
        "ecfr_api": {
            "date": latest_date,
            "title": title,
            "part": hierarchy.get(
                "part"
            ),
            "subpart": hierarchy.get(
                "subpart"
            ),
            "section": hierarchy.get(
                "section"
            ),
            "api_url": response.url,
        },
    }


# ==================================================
# SOURCE TYPE DETECTION
# ==================================================


def is_ecfr_source(
    source,
):
    url = str(
        source.get(
            "url",
            ""
        )
    ).lower()

    agency = str(
        source.get(
            "agency",
            ""
        )
    ).lower()

    return (
        "ecfr.gov" in url
        or "ecfr" in agency
    )


# ==================================================
# INGEST SINGLE SOURCE
# ==================================================


def ingest_source(
    source,
):
    source_type = str(
        source.get(
            "type",
            "html",
        )
    ).lower()

    if is_ecfr_source(
        source
    ):

        extracted = ingest_ecfr(
            source
        )

    elif source_type == "pdf":

        extracted = ingest_pdf(
            source
        )

    else:

        extracted = ingest_html(
            source
        )

    metadata = {
        "id": source.get(
            "id"
        ),
        "agency": source.get(
            "agency"
        ),
        "title": source.get(
            "title"
        ),
        "stage": source.get(
            "stage"
        ),
        "topic": source.get(
            "topic"
        ),
        "type": source.get(
            "type"
        ),
        "url": source.get(
            "url"
        ),

        # ------------------------------------------
        # Source policy metadata
        # ------------------------------------------

        "authority_type": source.get(
            "authority_type",
            "OFFICIAL_GUIDANCE",
        ),

        "source_status": source.get(
            "source_status",
            "CURRENT",
        ),

        "publication_date": source.get(
            "publication_date"
        ),

        "effective_date": source.get(
            "effective_date"
        ),

        "last_reviewed_date": source.get(
            "last_reviewed_date"
        ),

        "supersedes": source.get(
            "supersedes",
            [],
        ),

        "superseded_by": source.get(
            "superseded_by"
        ),
    }

    result = {
        "metadata": metadata,
        "text": extracted.get(
            "text",
            "",
        ),
        "pages": extracted.get(
            "pages",
            [],
        ),
    }

    if extracted.get(
        "ecfr_api"
    ):

        result[
            "ecfr_api"
        ] = extracted[
            "ecfr_api"
        ]

    return result


# ==================================================
# SAVE RAW DOCUMENT
# ==================================================


def save_raw_document(
    source_id,
    document,
):

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RAW_DIR
        / f"{source_id}.json"
    )

    output_path.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


# ==================================================
# VALIDATION
# ==================================================


def validate_ingested_document(
    source,
    document,
):
    text = document.get(
        "text",
        ""
    )

    if not text.strip():

        raise ValueError(
            "Extracted document text is empty."
        )

    if len(text) < 100:

        raise ValueError(
            f"Extracted text is suspiciously short: "
            f"{len(text)} characters."
        )

    # Detect the exact eCFR navigation-shell failure
    # that caused our cap-gap corpus bug.
    if is_ecfr_source(
        source
    ):

        navigation_shell = (
            "Navigate by entering citations or phrases"
        )

        if (
            navigation_shell.lower()
            in text.lower()
        ):

            raise ValueError(
                "eCFR navigation shell detected "
                "instead of regulation content."
            )


# ==================================================
# INGEST CORPUS
# ==================================================


def ingest_corpus():

    sources = load_sources()

    success = 0
    failed = 0

    print(
        "=" * 70
    )

    print(
        "KK-GPT CORPUS INGESTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Sources: {len(sources)}"
    )

    for index, source in enumerate(
        sources,
        start=1,
    ):

        source_id = source.get(
            "id",
            f"source_{index}",
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{index}/{len(sources)}] "
            f"{source_id}"
        )

        print(
            f"Title: "
            f"{source.get('title')}"
        )

        print(
            f"URL: "
            f"{source.get('url')}"
        )

        try:

            document = ingest_source(
                source
            )

            validate_ingested_document(
                source,
                document,
            )

            output_path = (
                save_raw_document(
                    source_id,
                    document,
                )
            )

            text_length = len(
                document.get(
                    "text",
                    ""
                )
            )

            print(
                f"SUCCESS"
            )

            print(
                f"Characters: "
                f"{text_length:,}"
            )

            print(
                f"Saved: "
                f"{output_path}"
            )

            success += 1

        except Exception as exc:

            print(
                f"FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed += 1

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Success: {success}"
    )

    print(
        f"Failed:  {failed}"
    )


# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":

    ingest_corpus()