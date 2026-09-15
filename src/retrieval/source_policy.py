from datetime import date, datetime


# ==================================================
# SOURCE AUTHORITY LEVELS
# ==================================================
#
# These are KK-Gpt internal ranking weights.
#
# They are NOT a legal statement that one source
# automatically overrides another in every context.
#
# The purpose is to help final evidence ranking
# prefer stronger primary authority when relevance
# is otherwise similar.
# ==================================================

AUTHORITY_LEVELS = {
    "REGULATION": 5,
    "POLICY_MANUAL": 4,
    "OFFICIAL_GUIDANCE": 3,
    "FORM_INSTRUCTIONS": 2,
    "ARCHIVED_GUIDANCE": 1,
}


DEFAULT_AUTHORITY_TYPE = "OFFICIAL_GUIDANCE"


# ==================================================
# AUTHORITY SCORE
# ==================================================


def get_authority_score(
    chunk: dict,
) -> float:
    """
    Return the internal KK-Gpt authority score.

    Expected metadata values:

        REGULATION
        POLICY_MANUAL
        OFFICIAL_GUIDANCE
        FORM_INSTRUCTIONS
        ARCHIVED_GUIDANCE

    Missing or unknown authority types safely
    default to OFFICIAL_GUIDANCE.
    """

    authority_type = chunk.get(
        "authority_type",
        DEFAULT_AUTHORITY_TYPE,
    )

    authority_type = str(
        authority_type
    ).upper()

    return float(
        AUTHORITY_LEVELS.get(
            authority_type,
            AUTHORITY_LEVELS[
                DEFAULT_AUTHORITY_TYPE
            ],
        )
    )


# ==================================================
# DATE PARSING
# ==================================================


def parse_date(
    value,
):
    """
    Parse YYYY-MM-DD metadata into a Python date.

    Returns None when:
    - the value is missing
    - the format is invalid
    """

    if not value:
        return None

    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    try:
        return datetime.strptime(
            str(value),
            "%Y-%m-%d",
        ).date()

    except ValueError:
        return None


# ==================================================
# SOURCE STATUS
# ==================================================


def is_source_active(
    chunk: dict,
) -> bool:
    """
    Only CURRENT sources are treated as active.

    These are inactive:

        SUPERSEDED
        ARCHIVED
        WITHDRAWN

    Missing source_status defaults to CURRENT.
    """

    status = str(
        chunk.get(
            "source_status",
            "CURRENT",
        )
    ).upper()

    return status == "CURRENT"


# ==================================================
# FRESHNESS SIGNAL
# ==================================================


def get_freshness_score(
    chunk: dict,
    today=None,
) -> float:
    """
    Return a small freshness signal.

    Freshness must never dominate:
    - semantic relevance
    - reranker relevance
    - legal/source authority

    Preferred metadata order:

        effective_date
        last_reviewed_date
        publication_date

    Approximate score range:
        0.2 -> very old
        0.5 -> unknown / neutral
        1.0 -> recent
    """

    if today is None:
        today = date.today()

    date_value = (
        chunk.get(
            "effective_date"
        )
        or chunk.get(
            "last_reviewed_date"
        )
        or chunk.get(
            "publication_date"
        )
    )

    source_date = parse_date(
        date_value
    )

    # Unknown date = neutral.
    if source_date is None:
        return 0.5

    age_days = (
        today - source_date
    ).days

    # Future effective/publication date.
    #
    # We deliberately keep this neutral rather than
    # treating it as current authority.
    if age_days < 0:
        return 0.5

    age_years = (
        age_days / 365.25
    )

    if age_years <= 1:
        return 1.0

    if age_years <= 3:
        return 0.8

    if age_years <= 5:
        return 0.6

    if age_years <= 10:
        return 0.4

    return 0.2


# ==================================================
# TEMPORAL STATUS
# ==================================================


def get_temporal_status(
    chunk: dict,
) -> str:
    """
    Normalize temporal/source status.

    Supported statuses:

        CURRENT
        SUPERSEDED
        ARCHIVED
        WITHDRAWN

    Unknown values fail conservatively to CURRENT
    for now because existing corpus metadata may
    omit the field.

    We can tighten this later once every source is
    fully versioned.
    """

    status = str(
        chunk.get(
            "source_status",
            "CURRENT",
        )
    ).upper()

    allowed = {
        "CURRENT",
        "SUPERSEDED",
        "ARCHIVED",
        "WITHDRAWN",
    }

    if status not in allowed:
        return "CURRENT"

    return status


# ==================================================
# SOURCE POLICY SIGNALS
# ==================================================


def get_source_policy_signals(
    chunk: dict,
) -> dict:
    """
    Return all source-quality signals used by
    authority-aware evidence ranking.
    """

    authority_type = str(
        chunk.get(
            "authority_type",
            DEFAULT_AUTHORITY_TYPE,
        )
    ).upper()

    authority_score = (
        get_authority_score(
            chunk
        )
    )

    freshness_score = (
        get_freshness_score(
            chunk
        )
    )

    source_status = (
        get_temporal_status(
            chunk
        )
    )

    active = is_source_active(
        chunk
    )

    return {
        "authority_type": (
            authority_type
        ),
        "authority_score": (
            authority_score
        ),
        "freshness_score": (
            freshness_score
        ),
        "source_status": (
            source_status
        ),
        "active": (
            active
        ),
        "publication_date": chunk.get(
            "publication_date"
        ),
        "effective_date": chunk.get(
            "effective_date"
        ),
        "last_reviewed_date": chunk.get(
            "last_reviewed_date"
        ),
        "supersedes": chunk.get(
            "supersedes",
            [],
        ),
        "superseded_by": chunk.get(
            "superseded_by"
        ),
    }


# ==================================================
# DEBUG / MANUAL TEST
# ==================================================


def main():

    current_regulation = {
        "agency": "DHS-eCFR",
        "document": "Example Regulation",
        "authority_type": "REGULATION",
        "source_status": "CURRENT",
        "last_reviewed_date": "2026-09-13",
    }

    archived_guidance = {
        "agency": "USCIS",
        "document": "Example Archived Guidance",
        "authority_type": "OFFICIAL_GUIDANCE",
        "source_status": "ARCHIVED",
        "publication_date": "2013-01-01",
    }

    print(
        "========================================"
    )

    print(
        "CURRENT REGULATION"
    )

    print(
        "========================================"
    )

    current_signals = (
        get_source_policy_signals(
            current_regulation
        )
    )

    for key, value in (
        current_signals.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "========================================"
    )

    print(
        "ARCHIVED GUIDANCE"
    )

    print(
        "========================================"
    )

    archived_signals = (
        get_source_policy_signals(
            archived_guidance
        )
    )

    for key, value in (
        archived_signals.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    assert (
        current_signals[
            "active"
        ]
        is True
    )

    assert (
        archived_signals[
            "active"
        ]
        is False
    )

    print(
        "M3.4 temporal status test: PASS"
    )


if __name__ == "__main__":
    main()