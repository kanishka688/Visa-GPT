from src.retrieval.source_policy import (
    get_source_policy_signals,
)


def main():

    current_regulation = {
        "authority_type": "REGULATION",
        "source_status": "CURRENT",
    }

    archived_guidance = {
        "authority_type": "OFFICIAL_GUIDANCE",
        "source_status": "ARCHIVED",
    }

    current = get_source_policy_signals(
        current_regulation
    )

    archived = get_source_policy_signals(
        archived_guidance
    )

    print(
        "Current regulation active:",
        current["active"],
    )

    print(
        "Archived guidance active:",
        archived["active"],
    )

    assert current["active"] is True

    assert archived["active"] is False

    print()
    print("M3.4 temporal status test: PASS")


if __name__ == "__main__":
    main()