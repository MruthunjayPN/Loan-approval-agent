"""
Phase 2 sanity check — an eyeball tool, not a pytest suite. Run from repo
root:

    python tests/sanity_check_phase2.py

Confirms two things visually: (1) Agent 1 actually catches a missing field
and an out-of-range value rather than passing them through, and (2) Agent
2's feature_vector lines up 1-for-1, in order, with model.pkl['feature_names'].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.validation_agent import run as run_validation
from agents.feature_engineering_agent import run as run_feature_engineering, load_model_artifact

CLEAN_APPLICATION = {
    "age": 34,
    "annual_income": 62000.0,
    "loan_amount": 15000.0,
    "credit_score": 710,
    "employment_type": "salaried",
    "years_employed": 6,
    "existing_loans": 1,
    "default_history": False,
}

MISSING_FIELD_APPLICATION = {k: v for k, v in CLEAN_APPLICATION.items() if k != "credit_score"}

OUT_OF_RANGE_APPLICATION = {**CLEAN_APPLICATION, "credit_score": 150, "age": 12}


def run_agent1(label: str, application: dict) -> dict:
    state = {"application": application}
    state.update(run_validation(state))
    print(f"--- {label} ---")
    print(f"validation_status: {state['validation_status']}")
    print(f"validation_errors: {state['validation_errors']}")
    print()
    return state


if __name__ == "__main__":
    run_agent1("clean application", CLEAN_APPLICATION)
    run_agent1("missing field (credit_score)", MISSING_FIELD_APPLICATION)
    run_agent1("out-of-range (credit_score=150, age=12)", OUT_OF_RANGE_APPLICATION)

    clean_state = run_agent1("clean application -> feeding into Agent 2", CLEAN_APPLICATION)
    clean_state.update(run_feature_engineering(clean_state))

    feature_names = load_model_artifact()["feature_names"]
    feature_vector = clean_state["feature_vector"]

    print("--- Agent 2 feature_vector vs model.pkl['feature_names'] ---")
    print(f"feature_version: {clean_state['feature_version']}")
    print(f"count match: {len(feature_vector)} vs {len(feature_names)}")
    for name, value in zip(feature_names, feature_vector):
        print(f"  {name:28s} = {value}")
