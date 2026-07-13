"""
Agent 1 — Input Validation.

Reads the raw application payload, writes validation_status /
validation_errors onto AgentState. Never touches feature_vector, scoring,
or SHAP — see ../phase0-task1-agent-contracts.md for the full contract this
implements. A FAILED status is the conditional short-circuit point graph.py
(Phase 4) routes on; this agent only has to produce a clean, structured
failure — it doesn't decide what happens next.
"""
from typing import Any

REQUIRED_FIELDS = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_type",
    "years_employed",
    "existing_loans",
    "default_history",
]

ALLOWED_EMPLOYMENT_TYPES = {"salaried", "self_employed", "unemployed"}

# credit_score bounds mirror data/generate_data.py's np.clip(300, 850) exactly.
# age bounds are a judgment call (see PROGRESS.md) — broader than the
# generator's 21-70 sampling range, so validation currently permits ages the
# model never saw in training.
MIN_AGE, MAX_AGE = 18, 100
MIN_CREDIT_SCORE, MAX_CREDIT_SCORE = 300, 850


def validate_application(application: dict[str, Any]) -> tuple[str, list[str]]:
    """Pure function: application dict in, (status, errors) out. No state
    dependency, so it's directly unit-testable without constructing AgentState."""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in application or application[field] is None:
            errors.append(f"missing required field: {field}")

    # Can't type/bound-check a field that isn't there — stop before the
    # per-field checks below would raise a KeyError on a missing key.
    if errors:
        return "FAILED", errors

    age = application["age"]
    annual_income = application["annual_income"]
    loan_amount = application["loan_amount"]
    credit_score = application["credit_score"]
    employment_type = application["employment_type"]
    years_employed = application["years_employed"]
    existing_loans = application["existing_loans"]
    default_history = application["default_history"]

    # isinstance(x, bool) excluded from the int/float checks below because
    # bool is a subclass of int in Python — True/False would otherwise
    # silently pass as valid ages, scores, etc.
    if not isinstance(age, int) or isinstance(age, bool):
        errors.append(f"age must be an int, got {type(age).__name__}")
    elif not (MIN_AGE <= age <= MAX_AGE):
        errors.append(f"age {age} out of bounds [{MIN_AGE}, {MAX_AGE}]")

    if not isinstance(annual_income, (int, float)) or isinstance(annual_income, bool):
        errors.append(f"annual_income must be numeric, got {type(annual_income).__name__}")
    elif annual_income <= 0:
        errors.append(f"annual_income must be positive, got {annual_income}")

    if not isinstance(loan_amount, (int, float)) or isinstance(loan_amount, bool):
        errors.append(f"loan_amount must be numeric, got {type(loan_amount).__name__}")
    elif loan_amount <= 0:
        errors.append(f"loan_amount must be positive, got {loan_amount}")

    if not isinstance(credit_score, int) or isinstance(credit_score, bool):
        errors.append(f"credit_score must be an int, got {type(credit_score).__name__}")
    elif not (MIN_CREDIT_SCORE <= credit_score <= MAX_CREDIT_SCORE):
        errors.append(
            f"credit_score {credit_score} out of bounds [{MIN_CREDIT_SCORE}, {MAX_CREDIT_SCORE}]"
        )

    if employment_type not in ALLOWED_EMPLOYMENT_TYPES:
        errors.append(
            f"employment_type {employment_type!r} not in {sorted(ALLOWED_EMPLOYMENT_TYPES)}"
        )

    if not isinstance(years_employed, int) or isinstance(years_employed, bool) or years_employed < 0:
        errors.append(f"years_employed must be a non-negative int, got {years_employed!r}")

    if not isinstance(existing_loans, int) or isinstance(existing_loans, bool) or existing_loans < 0:
        errors.append(f"existing_loans must be a non-negative int, got {existing_loans!r}")

    if not isinstance(default_history, bool):
        errors.append(f"default_history must be a bool, got {type(default_history).__name__}")

    return ("FAILED" if errors else "PASSED"), errors


def run(state: dict) -> dict:
    """LangGraph-node-style entry point: reads state['application'], returns
    only the keys this agent owns. Callers merge this into AgentState —
    application and every other existing key is left untouched (append-only)."""
    status, errors = validate_application(state["application"])
    return {"validation_status": status, "validation_errors": errors}
