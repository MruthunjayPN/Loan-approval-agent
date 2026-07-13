"""
Phase 3 sanity check — an eyeball tool, not a pytest suite. Run from repo
root:

    python tests/sanity_check_phase3.py

Runs 5 deliberately different synthetic applicants through Agents 1->2->3->4
end to end and prints probability + top SHAP drivers for each. Per the
brief: if a rejected high-risk applicant's SHAP output doesn't show low
credit_score / high loan_to_income_ratio as negative contributors, the bug
is in Phase 1's data generation, not here — this script does not adjust
anything to compensate, it just surfaces the numbers for a human to judge.

Also runs one negative test against Agent 4's SHAP-vs-probability
divergence check (see test_shap_divergence_check_fires below) — silence
under normal inputs doesn't prove that guard works, since SHAP values sum
to output-baseline by construction (the Shapley "efficiency" property).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.validation_agent import run as run_validation
from agents.feature_engineering_agent import run as run_feature_engineering
from agents.scoring_agent import run as run_scoring
from agents.shap_agent import run as run_shap, explain_prediction

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
    # NOT actually ambiguous -- moderate credit_score (620) but no default
    # history, and default_history carries the single largest risk weight
    # (0.35, see PROGRESS.md decisions log) -- it dominates and produces a
    # confident approve (0.7978). Kept as a case in its own right: "moderate
    # credit, no default -> confidently approved" is itself a useful,
    # honestly-labeled finding, not something to retune until it fits a
    # preconceived "borderline" label.
    "moderate credit, no default history -> confidently approved": {
        "age": 35,
        "annual_income": 55000.0,
        "loan_amount": 22000.0,
        "credit_score": 620,
        "employment_type": "self_employed",
        "years_employed": 3,
        "existing_loans": 2,
        "default_history": False,
    },
    # Genuinely ambiguous: found by empirically searching (not solved
    # analytically -- generate_data.py's normalize() is population-relative,
    # so the risk formula can't be inverted for a single applicant). Keeps
    # default_history=False, pushes credit_score down near the R1 floor
    # (580) and existing_loans up, landing at 0.5377 -- solidly inside R3's
    # 0.4-0.6 Refer zone, unlike the case above.
    "genuinely ambiguous (near R3 Refer zone, no default history)": {
        "age": 45,
        "annual_income": 42000.0,
        "loan_amount": 35000.0,
        "credit_score": 595,
        "employment_type": "salaried",
        "years_employed": 3,
        "existing_loans": 4,
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


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def test_shap_divergence_check_fires() -> None:
    """Negative test: feed explain_prediction() a deliberately wrong
    approval_probability and confirm the divergence warning actually logs.
    Without this, 'no warnings fired across 5 normal applicants' only shows
    the sigmoid conversion math is correct -- it never exercises the path
    that's supposed to catch a real mismatch."""
    clean_low_risk = APPLICANTS["clean low-risk"]
    state = run_pipeline(clean_low_risk)
    real_probability = state["approval_probability"]  # ~0.93
    wrong_probability = 0.10

    handler = _CaptureHandler()
    shap_logger = logging.getLogger("agents.shap_agent")
    shap_logger.addHandler(handler)
    shap_logger.setLevel(logging.WARNING)
    try:
        explain_prediction(state["feature_vector"], wrong_probability)
    finally:
        shap_logger.removeHandler(handler)

    fired = any("diverges" in msg for msg in handler.records)
    print("=== negative test: SHAP-vs-Agent-3 divergence warning ===")
    print(f"real approval_probability: {real_probability:.4f}, passed in: {wrong_probability}")
    print(f"warning fired: {fired}")
    for msg in handler.records:
        print(f"  captured: {msg}")
    print()
    assert fired, "expected divergence warning did not fire -- the guard doesn't work"


if __name__ == "__main__":
    test_shap_divergence_check_fires()


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
