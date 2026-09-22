import json

from src.core.llm_client import call_llm


# ==================================================
# POLICY CATEGORIES
# ==================================================

ALLOWED_CATEGORIES = {
    "OFFICIAL_FACT",
    "CASE_PREDICTION",
    "FUTURE_PREDICTION",
    "RECOMMENDATION",
    "PERSONAL_DECISION",
    "LOCAL_SERVICE",
    "UNKNOWN",
}


# ==================================================
# QUERY POLICY CLASSIFIER
# ==================================================


def classify_query(
    question: str,
) -> dict:
    """
    Classify a user question before sending it
    into the official immigration RAG pipeline.

    Important:
    This classifier does NOT answer immigration
    questions.

    It only decides whether the question should
    be routed to official-source retrieval.
    """

    system_prompt = """
You are the query-routing classifier for KK-GPT,
a U.S. immigration information system.

Your ONLY task is to classify the user's intent.

Do NOT answer the immigration question.
Do NOT provide immigration facts.
Do NOT give legal advice.

Choose exactly one category:

OFFICIAL_FACT
- The user asks about an immigration rule,
  requirement, definition, consequence, status,
  process, form, eligibility requirement,
  government procedure, deadline, or legally
  available immigration option.
- Questions may include personal circumstances
  while still asking what the official rule says.
- These questions should be routed to official
  government evidence.

CASE_PREDICTION
- The user asks whether USCIS, DOL, DOS, or
  another agency will approve, deny, reject,
  issue, select, certify, or decide their
  particular case.
- This includes asking whether a specific
  petition is likely to succeed.

FUTURE_PREDICTION
- The user asks about something not currently
  knowable from existing official rules.
- Examples:
  future Visa Bulletin movement,
  future lottery odds,
  future policy changes,
  future processing trends.

RECOMMENDATION
- The user asks which school, employer,
  attorney, consultancy, strategy, service,
  or option is best, safest, easiest,
  fastest, or preferable.

PERSONAL_DECISION
- The user asks the system to make a personal
  career or life decision.
- Examples:
  whether to quit,
  transfer employers,
  accept an offer,
  travel,
  pay for premium processing.

LOCAL_SERVICE
- The user asks to locate a lawyer,
  business, provider, school, or service
  near them.

UNKNOWN
- The intent does not clearly fit another
  category.

IMPORTANT DISTINCTIONS:

"What immigration options can exist after
H-1B employment ends?"
= OFFICIAL_FACT

This asks what options the law or official
rules make available.

"Which option should I choose after losing
my H-1B job?"
= RECOMMENDATION or PERSONAL_DECISION

This asks the system to select an option.

"What happens to an approved I-140 if I lose
my H-1B job?"
= OFFICIAL_FACT

This asks about the legal consequence of
an event.

"Will USCIS approve my I-140?"
= CASE_PREDICTION

This asks for the outcome of a particular case.

"Does losing my H-1B job immediately cancel
an approved I-140?"
= OFFICIAL_FACT

This asks about an immigration rule,
not whether an agency will approve a case.

"What happens if I lose my job while on H-1B?"
= OFFICIAL_FACT

"What is the H-1B grace period?"
= OFFICIAL_FACT

"What is Day 1 CPT?"
= OFFICIAL_FACT

"Can CPT start before DSO authorization?"
= OFFICIAL_FACT

"Which Day 1 CPT university is safest?"
= RECOMMENDATION

"Will USCIS approve my H-1B petition?"
= CASE_PREDICTION

"What will EB-2 India reach next year?"
= FUTURE_PREDICTION

"Should I quit my H-1B employer?"
= PERSONAL_DECISION

Return JSON only.

Required format:

{
  "category": "OFFICIAL_FACT",
  "route_to_rag": true
}

route_to_rag MUST be true only when
category is OFFICIAL_FACT.

Do not wrap the JSON in markdown fences.
Do not include any explanation before or after it.
"""

    user_prompt = f"""
Classify this question:

{question}
"""

    try:

        raw_content = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=100,
        )

    except Exception:

        # Fail closed if the model provider,
        # network, API, or local inference fails.
        return {
            "category": "UNKNOWN",
            "route_to_rag": False,
        }

    # ----------------------------------------------
    # PARSE JSON
    # ----------------------------------------------

    try:

        result = json.loads(
            raw_content
        )

    except json.JSONDecodeError:

        return {
            "category": "UNKNOWN",
            "route_to_rag": False,
        }

    category = result.get(
        "category",
        "UNKNOWN",
    )

    # ----------------------------------------------
    # FAIL CLOSED
    # ----------------------------------------------

    if category not in ALLOWED_CATEGORIES:

        category = "UNKNOWN"

    # Never trust the model's route_to_rag value.
    #
    # Routing is derived deterministically from
    # the validated category.
    route_to_rag = (
        category
        == "OFFICIAL_FACT"
    )

    return {
        "category": category,
        "route_to_rag": route_to_rag,
    }


# ==================================================
# COMMAND LINE TEST
# ==================================================


def main():

    question = input(
        "Ask KK-GPT: "
    ).strip()

    result = classify_query(
        question
    )

    print()

    print(
        "========================================"
    )

    print(
        "QUERY POLICY"
    )

    print(
        "========================================"
    )

    print(
        f"Category: "
        f"{result['category']}"
    )

    print(
        f"Route to RAG: "
        f"{result['route_to_rag']}"
    )


if __name__ == "__main__":
    main()