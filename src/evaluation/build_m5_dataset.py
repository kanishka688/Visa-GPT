from pathlib import Path
import json


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

BASE_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions_gold_draft.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/evaluation/questions_gold_v1.json"
)


# ==================================================
# NEW M5 QUESTIONS
# ==================================================

NEW_QUESTIONS = [

    # ==================================================
    # CPT / F-1
    # ==================================================

    {
        "id": "f1_007",
        "question": (
            "Do F-1 students normally need to complete "
            "one academic year before becoming eligible "
            "for CPT?"
        ),
        "stage": "F-1",
        "topic": "CPT",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2",
            "sevp_practical_training_overview"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "f1_008",
        "question": (
            "Does part-time CPT count toward the "
            "12-month full-time CPT limit that affects OPT?"
        ),
        "stage": "F-1",
        "topic": "CPT",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2",
            "sevp_practical_training_overview"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "f1_009",
        "question": (
            "Who signs off on CPT for an F-1 student?"
        ),
        "stage": "F-1",
        "topic": "CPT",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2",
            "sevp_practical_training_overview"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "f1_010",
        "question": (
            "Can I begin working on CPT before my DSO "
            "authorizes it in SEVIS?"
        ),
        "stage": "F-1",
        "topic": "CPT",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2",
            "sevp_practical_training_overview",
            "sevp_cpt_immediate_graduate_guidance"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # OPT
    # ==================================================

    {
        "id": "opt_004",
        "question": (
            "Who authorizes OPT employment for an "
            "F-1 student?"
        ),
        "stage": "F-1",
        "topic": "OPT",
        "answerable": True,
        "expected_source_ids": [
            "uscis_opt_f1_students",
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "opt_005",
        "question": (
            "Does using pre-completion OPT reduce the "
            "amount of post-completion OPT available?"
        ),
        "stage": "F-1",
        "topic": "OPT",
        "answerable": True,
        "expected_source_ids": [
            "uscis_opt_f1_students",
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "opt_006",
        "question": (
            "Does OPT employment need to be related "
            "to the student's field of study?"
        ),
        "stage": "F-1",
        "topic": "OPT",
        "answerable": True,
        "expected_source_ids": [
            "uscis_opt_f1_students",
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # STEM OPT
    # ==================================================

    {
        "id": "stem_003",
        "question": (
            "How long is the STEM OPT extension?"
        ),
        "stage": "F-1",
        "topic": "STEM OPT",
        "answerable": True,
        "expected_source_ids": [
            "uscis_stem_opt_policy",
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "stem_004",
        "question": (
            "Is E-Verify mandatory for an employer "
            "hiring a student on STEM OPT?"
        ),
        "stage": "F-1",
        "topic": "STEM OPT",
        "answerable": True,
        "expected_source_ids": [
            "uscis_i765_instructions",
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "stem_005",
        "question": (
            "What is Form I-983 used for in the "
            "STEM OPT process?"
        ),
        "stage": "F-1",
        "topic": "STEM OPT",
        "answerable": True,
        "expected_source_ids": [
            "uscis_stem_opt_policy"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # H-1B
    # ==================================================

    {
        "id": "h1b_005",
        "question": (
            "What is the maximum discretionary grace "
            "period after H-1B employment ends?"
        ),
        "stage": "H-1B",
        "topic": "H-1B grace period",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_1",
            "uscis_h1b_termination_options"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "h1b_006",
        "question": (
            "What does H-1B portability mean?"
        ),
        "stage": "H-1B",
        "topic": "H-1B portability",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "h1b_007",
        "question": (
            "When can an eligible H-1B worker start "
            "working for a new employer under portability?"
        ),
        "stage": "H-1B",
        "topic": "H-1B portability",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "h1b_008",
        "question": (
            "Does an H-1B employer need a Labor "
            "Condition Application?"
        ),
        "stage": "H-1B",
        "topic": "H-1B LCA",
        "answerable": True,
        "expected_source_ids": [
            "dol_h1b_lca",
            "ecfr_20cfr_655_subpart_h"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "h1b_009",
        "question": (
            "I was laid off on H-1B. What official "
            "immigration options may exist after "
            "termination of employment?"
        ),
        "stage": "H-1B",
        "topic": "H-1B termination",
        "answerable": True,
        "expected_source_ids": [
            "uscis_h1b_termination_options",
            "ecfr_8cfr_214_1"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # CAP-GAP
    # ==================================================

    {
        "id": "h1b_010",
        "question": (
            "What does the F-1 cap-gap provision "
            "automatically extend?"
        ),
        "stage": "F-1 H-1B",
        "topic": "cap-gap",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_8cfr_214_2"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # H-4
    # ==================================================

    {
        "id": "h4_002",
        "question": (
            "Are all H-4 spouses automatically "
            "authorized to work?"
        ),
        "stage": "H-4",
        "topic": "H-4 EAD",
        "answerable": True,
        "expected_source_ids": [
            "uscis_h4_employment_authorization",
            "ecfr_8cfr_274a_12"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # PERM / PWD
    # ==================================================

    {
        "id": "perm_004",
        "question": (
            "Who issues the prevailing wage "
            "determination used in the PERM process?"
        ),
        "stage": "PERM",
        "topic": "Prevailing Wage",
        "answerable": True,
        "expected_source_ids": [
            "dol_prevailing_wages",
            "ecfr_20cfr_656"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "perm_005",
        "question": (
            "What role does the prevailing wage "
            "determination play in PERM?"
        ),
        "stage": "PERM",
        "topic": "Prevailing Wage",
        "answerable": True,
        "expected_source_ids": [
            "dol_prevailing_wages",
            "ecfr_20cfr_656"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "perm_006",
        "question": (
            "What is ETA Form 9089 used for?"
        ),
        "stage": "PERM",
        "topic": "PERM ETA-9089",
        "answerable": True,
        "expected_source_ids": [
            "ecfr_20cfr_656"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # I-140
    # ==================================================

    {
        "id": "i140_004",
        "question": (
            "Who generally files Form I-140?"
        ),
        "stage": "I-140",
        "topic": "I-140",
        "answerable": True,
        "expected_source_ids": [
            "uscis_i140",
            "uscis_i140_instructions"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "i140_005",
        "question": (
            "What employment-based immigrant "
            "categories are covered by Form I-140?"
        ),
        "stage": "I-140",
        "topic": "I-140",
        "answerable": True,
        "expected_source_ids": [
            "uscis_i140",
            "ecfr_8cfr_204_5"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    {
        "id": "i140_006",
        "question": (
            "How is a priority date determined for "
            "an employment-based immigrant petition?"
        ),
        "stage": "I-140",
        "topic": "Priority Date",
        "answerable": True,
        "expected_source_ids": [
            "uscis_priority_dates",
            "ecfr_8cfr_204_5"
        ],
        "expected_policy_category": "OFFICIAL_FACT"
    },

    # ==================================================
    # UNANSWERABLE / ROUTING TESTS
    # ==================================================

    {
        "id": "negative_009",
        "question": (
            "Will my H-1B transfer definitely be approved?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "case prediction",
        "answerable": False,
        "expected_policy_category": "CASE_PREDICTION"
    },

    {
        "id": "negative_010",
        "question": (
            "Which Day 1 CPT university has the lowest "
            "chance of an RFE?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "recommendation",
        "answerable": False,
        "expected_policy_category": "RECOMMENDATION"
    },

    {
        "id": "negative_011",
        "question": (
            "Will the H-1B lottery become easier next year?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "future prediction",
        "answerable": False,
        "expected_policy_category": "FUTURE_PREDICTION"
    },

    {
        "id": "negative_012",
        "question": (
            "Should I switch from OPT to Day 1 CPT?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "personal decision",
        "answerable": False,
        "expected_policy_category": "PERSONAL_DECISION"
    },

    {
        "id": "negative_013",
        "question": (
            "Find me the cheapest immigration attorney "
            "near my apartment."
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "local service",
        "answerable": False,
        "expected_policy_category": "LOCAL_SERVICE"
    },

    {
        "id": "negative_014",
        "question": (
            "What will the EB-2 India final action "
            "date be in 2028?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "future prediction",
        "answerable": False,
        "expected_policy_category": "FUTURE_PREDICTION"
    },

    {
        "id": "negative_015",
        "question": (
            "Which employer is safest for getting "
            "my H-1B approved?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "recommendation",
        "answerable": False,
        "expected_policy_category": "RECOMMENDATION"
    },

    {
        "id": "negative_016",
        "question": (
            "Based on my profile, will USCIS send "
            "me an RFE?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "case prediction",
        "answerable": False,
        "expected_policy_category": "CASE_PREDICTION"
    },

    {
        "id": "negative_017",
        "question": (
            "Should I resign before my I-140 is approved?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "personal decision",
        "answerable": False,
        "expected_policy_category": "PERSONAL_DECISION"
    },

    {
        "id": "negative_018",
        "question": (
            "Which consulting company has the easiest "
            "PERM process?"
        ),
        "stage": "OUT_OF_SCOPE",
        "topic": "recommendation",
        "answerable": False,
        "expected_policy_category": "RECOMMENDATION"
    }
]


# ==================================================
# BUILD DATASET
# ==================================================


def build_dataset():

    base = json.loads(
        BASE_PATH.read_text(
            encoding="utf-8"
        )
    )

    # Remove temporary candidate-source helper fields.
    cleaned_base = []

    for item in base:

        cleaned = {
            key: value
            for key, value in item.items()
            if key != "_candidate_sources"
        }

        cleaned_base.append(
            cleaned
        )

    existing_ids = {
        item["id"]
        for item in cleaned_base
    }

    for item in NEW_QUESTIONS:

        if item["id"] in existing_ids:

            raise ValueError(
                f"Duplicate question ID: "
                f"{item['id']}"
            )

    dataset = (
        cleaned_base
        + NEW_QUESTIONS
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            dataset,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    answerable = sum(
        1
        for item in dataset
        if item.get(
            "answerable",
            True,
        )
    )

    unanswerable = (
        len(dataset)
        - answerable
    )

    print(
        "=" * 60
    )

    print(
        "M5 GOLD DATASET CREATED"
    )

    print(
        "=" * 60
    )

    print(
        f"Total:        {len(dataset)}"
    )

    print(
        f"Answerable:   {answerable}"
    )

    print(
        f"Unanswerable: {unanswerable}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":

    build_dataset()