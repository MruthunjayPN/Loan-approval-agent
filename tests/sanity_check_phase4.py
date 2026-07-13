"""
Phase 4 sanity check — an eyeball tool, not a pytest suite. Run from repo
root:

    python tests/sanity_check_phase4.py

Runs the full compiled graph (validation -> feature_engineering -> scoring
-> shap -> policy) against:
  - the 5 Phase 3 applicants, now carried through to final_decision
  - one deliberately invalid application, confirming it short-circuits at
    validation and never reaches feature_engineering/scoring/shap/policy
  - one genuinely R3-Refer-zone applicant (loan_to_income_ratio <= 0.45,
    so R2 never fires; probability lands between 0.40 and 0.75)

Note: the Phase 3 "genuinely ambiguous" applicant (credit_score=595,
loan_to_income_ratio=0.833) triggers R2, not R3 -- flagged during Phase 4
planning. Kept here as the R2 test case; a separate fixture below tests R3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import run_application

APPLICANTS = {
    "clean low-risk (expect R3 Approve)": {
        "age": 40,
        "annual_income": 90000.0,
        "loan_amount": 15000.0,
        "credit_score": 780,
        "employment_type": "salaried",
        "years_employed": 15,
        "existing_loans": 0,
        "default_history": False,
    },
    "clean high-risk (expect R1 Reject via default_history)": {
        "age": 28,
        "annual_income": 30000.0,
        "loan_amount": 25000.0,
        "credit_score": 520,
        "employment_type": "unemployed",
        "years_employed": 0,
        "existing_loans": 4,
        "default_history": True,
    },
    "moderate credit, no default history (expect R3 Approve)": {
        "age": 35,
        "annual_income": 55000.0,
        "loan_amount": 22000.0,
        "credit_score": 620,
        "employment_type": "self_employed",
        "years_employed": 3,
        "existing_loans": 2,
        "default_history": False,
    },
    "high LTI, no default history (expect R2 Refer, not R3)": {
        "age": 45,
        "annual_income": 42000.0,
        "loan_amount": 35000.0,
        "credit_score": 595,
        "employment_type": "salaried",
        "years_employed": 3,
        "existing_loans": 4,
        "default_history": False,
    },
    "genuinely R3 Refer-zone (LTI<=0.45, prob in (0.40, 0.75))": {
        "age": 30,
        "annual_income": 45000.0,
        "loan_amount": 20000.0,
        "credit_score": 583,
        "employment_type": "unemployed",
        "years_employed": 0,
        "existing_loans": 4,
        "default_history": False,
    },
    # Guessed "R3 Approve" before running this -- wrong. Actual probability
    # (0.7069) lands under the 0.75 auto-approve threshold, so this is an
    # R3 Refer, not an Approve. Left as-is and relabeled rather than tuned,
    # per the same "don't retune a fixture to fit a preconception" rule
    # from Phase 3 -- this is a legitimate, useful finding on its own:
    # credit_score sitting exactly at the R1 floor (580) is not enough on
    # its own to clear R3's approve bar, even with default_history=False.
    "edge case near validation boundary (age=70, credit_score=580 -> actually R3 Refer, not Approve)": {
        "age": 70,
        "annual_income": 40000.0,
        "loan_amount": 18000.0,
        "credit_score": 580,
        "employment_type": "salaried",
        "years_employed": 45,
        "existing_loans": 2,
        "default_history": False,
    },
    "INVALID: missing credit_score (expect short-circuit at validation)": {
        "age": 34,
        "annual_income": 62000.0,
        "loan_amount": 15000.0,
        "employment_type": "salaried",
        "years_employed": 6,
        "existing_loans": 1,
        "default_history": False,
    },
}


if __name__ == "__main__":
    for label, application in APPLICANTS.items():
        state = run_application(application)
        print(f"=== {label} ===")

        if state["validation_status"] != "PASSED":
            print(f"validation_status: FAILED -- {state['validation_errors']}")
            never_ran = [k for k in ("feature_vector", "approval_probability", "shap_top_drivers") if k in state]
            print(f"short-circuited correctly (no downstream keys present): {not never_ran}")
        else:
            print(f"approval_probability: {state['approval_probability']:.4f}")

        print(f"final_decision: {state['final_decision']}")
        print(f"triggered_rules: {state['triggered_rules']}")
        print(f"reviewer_required: {state['reviewer_required']}")
        print(f"decision_reason: {state['decision_reason']}")
        print()
