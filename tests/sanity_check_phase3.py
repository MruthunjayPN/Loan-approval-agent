"""
Phase 3 sanity check — an eyeball tool, not a pytest suite. Run from repo
root:

    python tests/sanity_check_phase3.py

Runs 4 deliberately different synthetic applicants through Agents 1->2->3->4
end to end and prints probability + top SHAP drivers for each. Per the
brief: if a rejected high-risk applicant's SHAP output doesn't show low
credit_score / high loan_to_income_ratio as negative contributors, the bug
is in Phase 1's data generation, not here — this script does not adjust
anything to compensate, it just surfaces the numbers for a human to judge.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.validation_agent import run as run_validation
from agents.feature_engineering_agent import run as run_feature_engineering
from agents.scoring_agent import run as run_scoring
from agents.shap_agent import run as run_shap

APPLICANTS = {
    "clean low-risk": {
        "age": 40,
        "annual_income": 90000.0,
        "loan_amount": 15000.0,
        "credit_score": 780,
        "employment_type": "salaried",
        "years_employed": 15,
        "existing_loans": 0,
        "default_history": False,
    },
    "clean high-risk": {
        "age": 28,
        "annual_income": 30000.0,
        "loan_amount": 25000.0,
        "credit_score": 520,
        "employment_type": "unemployed",
        "years_employed": 0,
        "existing_loans": 4,
        "default_history": True,
    },
    "borderline / ambiguous": {
        "age": 35,
        "annual_income": 55000.0,
        "loan_amount": 22000.0,
        "credit_score": 620,
        "employment_type": "self_employed",
        "years_employed": 3,
        "existing_loans": 2,
        "default_history": False,
    },
    "edge case near validation boundary (age=70, credit_score=580)": {
        "age": 70,
        "annual_income": 40000.0,
        "loan_amount": 18000.0,
        "credit_score": 580,
        "employment_type": "salaried",
        "years_employed": 45,
        "existing_loans": 2,
        "default_history": False,
    },
}


def run_pipeline(application: dict) -> dict:
    state = {"application": application}
    state.update(run_validation(state))
    if state["validation_status"] != "PASSED":
        return state
    state.update(run_feature_engineering(state))
    state.update(run_scoring(state))
    state.update(run_shap(state))
    return state


if __name__ == "__main__":
    for label, application in APPLICANTS.items():
        state = run_pipeline(application)
        print(f"=== {label} ===")
        print(f"application: {application}")

        if state["validation_status"] != "PASSED":
            print(f"validation FAILED: {state['validation_errors']}")
            print()
            continue

        print(f"approval_probability: {state['approval_probability']:.4f}")
        print(f"prediction_class (raw, 0.5 cutoff, NOT the policy decision): {state['prediction_class']}")
        print(f"model_version: {state['model_version']}  inference_latency: {state['inference_latency']:.5f}s")
        print(f"explanation_summary: {state['explanation_summary']}")
        print("shap_top_drivers:")
        for driver in state["shap_top_drivers"]:
            print(
                f"  {driver['feature']:28s} contribution={driver['contribution']:+.4f}  "
                f"direction={driver['direction']}"
            )
        print()
