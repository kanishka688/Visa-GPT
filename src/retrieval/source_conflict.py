from datetime import date

from src.retrieval.source_policy import (
    get_source_policy_signals,
    parse_date,
)


# ==================================================
# SOURCE ID
# ==================================================


def get_source_id(
    source: dict,
):
    return (
        source.get("source_id")
        or source.get("id")
    )


# ==================================================
# EXPLICIT SUPERSESSION
# ==================================================


def explicitly_supersedes(
    newer: dict,
    older: dict,
) -> bool:
    """
    Return True when newer metadata explicitly says
    it supersedes the older source.
    """

    newer_id = get_source_id(
        newer
    )

    older_id = get_source_id(
        older
    )

    if not newer_id or not older_id:
        return False

    supersedes = newer.get(
        "supersedes",
        [],
    ) or []

    return older_id in supersedes


# ==================================================
# BEST SOURCE DATE
# ==================================================


def get_best_source_date(
    source: dict,
):
    """
    Date used only as a tie breaker.

    Preference:
        effective_date
        last_reviewed_date
        publication_date
    """

    value = (
        source.get("effective_date")
        or source.get("last_reviewed_date")
        or source.get("publication_date")
    )

    return parse_date(
        value
    )


# ==================================================
# SOURCE COMPARISON
# ==================================================


def choose_preferred_source(
    source_a: dict,
    source_b: dict,
) -> dict:
    """
    Choose which source should be preferred when
    evidence conflicts.

    Priority:

    1. Explicit supersession
    2. Active/current status
    3. Authority level
    4. Newer effective/review/publication date
    5. No preference
    """

    # ------------------------------------------------
    # 1. EXPLICIT VERSION RELATIONSHIP
    # ------------------------------------------------

    if explicitly_supersedes(
        source_a,
        source_b,
    ):
        return {
            "preferred": "A",
            "reason": "A explicitly supersedes B",
        }

    if explicitly_supersedes(
        source_b,
        source_a,
    ):
        return {
            "preferred": "B",
            "reason": "B explicitly supersedes A",
        }

    # ------------------------------------------------
    # 2. SOURCE STATUS
    # ------------------------------------------------

    signals_a = (
        get_source_policy_signals(
            source_a
        )
    )

    signals_b = (
        get_source_policy_signals(
            source_b
        )
    )

    if (
        signals_a["active"]
        and not signals_b["active"]
    ):
        return {
            "preferred": "A",
            "reason": (
                "A is current while B is inactive"
            ),
        }

    if (
        signals_b["active"]
        and not signals_a["active"]
    ):
        return {
            "preferred": "B",
            "reason": (
                "B is current while A is inactive"
            ),
        }

    # ------------------------------------------------
    # 3. AUTHORITY
    # ------------------------------------------------

    if (
        signals_a["authority_score"]
        > signals_b["authority_score"]
    ):
        return {
            "preferred": "A",
            "reason": (
                "A has higher source authority"
            ),
        }

    if (
        signals_b["authority_score"]
        > signals_a["authority_score"]
    ):
        return {
            "preferred": "B",
            "reason": (
                "B has higher source authority"
            ),
        }

    # ------------------------------------------------
    # 4. DATE
    # ------------------------------------------------

    date_a = get_best_source_date(
        source_a
    )

    date_b = get_best_source_date(
        source_b
    )

    if (
        date_a is not None
        and date_b is not None
    ):

        if date_a > date_b:

            return {
                "preferred": "A",
                "reason": (
                    "A has the newer applicable date"
                ),
            }

        if date_b > date_a:

            return {
                "preferred": "B",
                "reason": (
                    "B has the newer applicable date"
                ),
            }

    # ------------------------------------------------
    # 5. NO DETERMINISTIC WINNER
    # ------------------------------------------------

    return {
        "preferred": None,
        "reason": (
            "No deterministic source preference"
        ),
    }


# ==================================================
# MANUAL TEST
# ==================================================


def main():

    regulation = {
        "source_id": "current_regulation",
        "authority_type": "REGULATION",
        "source_status": "CURRENT",
        "last_reviewed_date": "2026-09-13",
    }

    old_guidance = {
        "source_id": "old_guidance",
        "authority_type": "OFFICIAL_GUIDANCE",
        "source_status": "ARCHIVED",
        "publication_date": "2013-01-01",
    }

    result = choose_preferred_source(
        regulation,
        old_guidance,
    )

    print(
        "========================================"
    )

    print(
        "SOURCE CONFLICT TEST"
    )

    print(
        "========================================"
    )

    print(
        "Preferred:",
        result["preferred"],
    )

    print(
        "Reason:",
        result["reason"],
    )

    assert (
        result["preferred"]
        == "A"
    )

    print()

    print(
        "M3.5 source conflict test: PASS"
    )


if __name__ == "__main__":
    main()