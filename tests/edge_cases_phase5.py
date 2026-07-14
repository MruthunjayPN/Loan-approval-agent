"""
Phase 5 edge-case check — an eyeball tool, not a pytest suite. Run from
repo root:

    python tests/edge_cases_phase5.py

Calls app.py's predict() directly (the actual Gradio callback function),
not graph.py -- this is deliberate: it confirms the UI layer's own call
path handles each case, and it's the only way to feed the deliberately
malformed cases (a missing field as None, credit_score as a string) at
all, since gr.Number's browser-side widget wouldn't let you type a string
into it. Calling predict() directly simulates what Gradio's own API
endpoint would accept from a non-browser caller, bypassing that
client-side constraint -- the same call path a malformed request could
reach in practice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import predict

BASE = {
    "age": 40,
    "annual_income": 60000.0,
    "loan_amount": 15000.0,
    "credit_score": 700,
    "employment_type": "salaried",
    "years_employed": 10,
    "existing_loans": 1,
    "default_history": False,
}

CASES = {
    "missing required field (credit_score=None)": {**BASE, "credit_score": None},
    "wrong type (credit_score as a string)": {**BASE, "credit_score": "seven hundred"},
    "credit_score exactly at 580 (R1 floor, should NOT reject)": {**BASE, "credit_score": 580},
    "credit_score exactly at 579 (one below floor, should reject)": {**BASE, "credit_score": 579},
    # loan_to_income_ratio = 27000/60000 = 0.45 exactly -- R2 is strict '>',
    # so this must NOT trigger R2.
    "loan_to_income_ratio exactly at 0.45 (R2 boundary, should NOT refer via R2)": {
        **BASE,
        "annual_income": 60000.0,
        "loan_amount": 27000.0,
    },
    # Below the MIN_ANNUAL_INCOME=8000 floor added this phase (matches
    # generate_data.py's np.clip(..., 8000, None) training floor) -- must
    # fail validation, not flow through to a nonsensical LTI ratio.
    "wildly out-of-range income (annual_income=5, should fail validation)": {**BASE, "annual_income": 5.0},
    # The clearest demonstration of why R2 exists separately from R3:
    # probability alone would clear R3's 0.75 auto-approve bar, but LTI
    # (0.6667) sits well above R2's 0.45 ceiling. Verified against the real
    # trained model in Phase 4's sanity check (probability=0.8680,
    # LTI=0.6667, Refer via R2) -- reproduced here through the UI layer too.
    "R2 genuinely overrides a would-be R3 auto-approve": {
        "age": 45,
        "annual_income": 60000.0,
        "loan_amount": 40000.0,
        "credit_score": 800,
        "employment_type": "salaried",
        "years_employed": 20,
        "existing_loans": 0,
        "default_history": False,
    },
}


if __name__ == "__main__":
    for label, application in CASES.items():
        print(f"=== {label} ===")
        (
            decision,
            decision_reason,
            approval_probability,
            triggered_rules,
            reviewer_required,
            validation_errors,
            shap_explanation,
        ) = predict(**application)
        print(f"decision: {decision}")
        print(f"decision_reason: {decision_reason}")
        print(f"approval_probability: {approval_probability}")
        print(f"triggered_rules: {triggered_rules}")
        print(f"reviewer_required: {reviewer_required}")
        print(f"validation_errors: {validation_errors}")
        print(f"shap_explanation: {shap_explanation}")
        print()
