"""
Agent 5 — Policy + Decision.

Implements ../phase0-task2-policy-rules.md exactly, in the evaluation order
specified there (stop on first match):
  R1 — hard reject floor: credit_score < 580 OR default_history == True
  R2 — affordability override: loan_to_income_ratio > 0.45 -> Refer
  R3 — ML threshold triage (only reached if R1/R2 didn't fire):
       probability >= 0.75 -> Approve, <= 0.40 -> Reject, else -> Refer

Deterministic rules always own the Approve/Reject/Refer outcome. The LLM
(Gemini) is invoked only when the outcome is Refer, and only to synthesize
decision_reason from structured inputs already computed by prior agents
(probability, SHAP drivers/summary, which rule almost fired and by how
much) — it never sees or reasons about rules it isn't given, and cannot
change the outcome, which is already decided before the LLM is called.

See ../phase0-task1-agent-contracts.md (Agent 5 section) for the full
contract this implements.
"""
import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from agents.feature_engineering_agent import load_model_artifact

load_dotenv()

R1_CREDIT_SCORE_FLOOR = 580
R2_LTI_CEILING = 0.45
R3_APPROVE_THRESHOLD = 0.75
R3_REJECT_THRESHOLD = 0.40

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    """Lazily constructed and cached at module level -- avoids requiring
    GEMINI_API_KEY to be set for Approve/Reject-only test runs that never
    reach the LLM step."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _extract_loan_to_income_ratio(feature_vector: list[float], feature_names: list[str]) -> float:
    """Reads the ratio Agent 2 already computed, by name -- never
    recomputed a third time from loan_amount/annual_income, matching the
    single-source-of-truth principle used for feature ordering."""
    return dict(zip(feature_names, feature_vector))["loan_to_income_ratio"]


def _synthesize_refer_reason(
    *,
    triggered_rule: str,
    approval_probability: float,
    loan_to_income_ratio: float,
    credit_score: float,
    shap_top_drivers: list[dict[str, Any]],
    explanation_summary: str | None,
) -> str:
    """LLM step — Refer outcomes only. The outcome ("Refer") is already
    decided by the caller before this runs; this function only writes the
    explanation text, and is explicitly told not to imply a decision."""
    near_misses = []
    r1_margin = credit_score - R1_CREDIT_SCORE_FLOOR
    if 0 <= r1_margin <= 30:
        near_misses.append(f"credit_score is only {r1_margin:.0f} points above the R1 hard-reject floor")
    r2_margin = R2_LTI_CEILING - loan_to_income_ratio
    if 0 <= r2_margin <= 0.05:
        near_misses.append(f"loan_to_income_ratio is only {r2_margin:.3f} below the R2 affordability ceiling")

    prompt = f"""You are writing a short, plain-English explanation for why a loan
application was referred to a human reviewer (not auto-approved or auto-rejected).

Facts (do not invent anything beyond these):
- Triggered rule: {triggered_rule}
- Model approval probability: {approval_probability:.4f}
- loan_to_income_ratio: {loan_to_income_ratio:.4f} (R2 refers anything above {R2_LTI_CEILING})
- R3 auto-decides only outside the [{R3_REJECT_THRESHOLD}, {R3_APPROVE_THRESHOLD}] probability range
- Near-miss notes: {"; ".join(near_misses) if near_misses else "none"}
- SHAP explanation of the model's own reasoning: {explanation_summary or "not available"}
- Top SHAP drivers: {shap_top_drivers}

Write 2-3 sentences explaining why this case is ambiguous enough to need human
review. Do not state or imply what the reviewer should decide -- that is not your role.
"""

    try:
        client = _get_gemini_client()
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as e:
        # LLM synthesis failing must not block the Refer decision itself --
        # same graceful-degradation principle Agent 4 uses for SHAP.
        return (
            f"[LLM reasoning unavailable ({e}) -- structured fallback] "
            f"Rule {triggered_rule} referred this application for human review "
            f"(approval_probability={approval_probability:.4f}, "
            f"loan_to_income_ratio={loan_to_income_ratio:.4f})."
        )


def evaluate_policy(
    application: dict[str, Any],
    feature_vector: list[float],
    feature_names: list[str],
    approval_probability: float,
    shap_top_drivers: list[dict[str, Any]],
    explanation_summary: str | None,
) -> dict[str, Any]:
    credit_score = application["credit_score"]
    default_history = application["default_history"]
    loan_to_income_ratio = _extract_loan_to_income_ratio(feature_vector, feature_names)

    # R1 — hard reject floor. Returning here means R2/R3 below never execute.
    if credit_score < R1_CREDIT_SCORE_FLOOR or default_history:
        reasons = []
        if credit_score < R1_CREDIT_SCORE_FLOOR:
            reasons.append(f"credit_score {credit_score} is below the {R1_CREDIT_SCORE_FLOOR} floor")
        if default_history:
            reasons.append("applicant has a prior default")
        return {
            "final_decision": "Rejected",
            "triggered_rules": ["R1"],
            "decision_reason": "Hard-rejected: " + "; ".join(reasons) + ".",
            "reviewer_required": False,
        }

    # R2 — affordability override. Returning here means R3 below never executes.
    if loan_to_income_ratio > R2_LTI_CEILING:
        decision_reason = _synthesize_refer_reason(
            triggered_rule="R2",
            approval_probability=approval_probability,
            loan_to_income_ratio=loan_to_income_ratio,
            credit_score=credit_score,
            shap_top_drivers=shap_top_drivers,
            explanation_summary=explanation_summary,
        )
        return {
            "final_decision": "Refer",
            "triggered_rules": ["R2"],
            "decision_reason": decision_reason,
            "reviewer_required": True,
        }

    # R3 — ML threshold triage (only reached if R1 and R2 didn't fire above)
    if approval_probability >= R3_APPROVE_THRESHOLD:
        return {
            "final_decision": "Approved",
            "triggered_rules": ["R3"],
            "decision_reason": (
                f"Approved: model probability {approval_probability:.4f} meets the "
                f"{R3_APPROVE_THRESHOLD} auto-approve threshold."
            ),
            "reviewer_required": False,
        }
    if approval_probability <= R3_REJECT_THRESHOLD:
        return {
            "final_decision": "Rejected",
            "triggered_rules": ["R3"],
            "decision_reason": (
                f"Rejected: model probability {approval_probability:.4f} is at or below the "
                f"{R3_REJECT_THRESHOLD} auto-reject threshold."
            ),
            "reviewer_required": False,
        }
    decision_reason = _synthesize_refer_reason(
        triggered_rule="R3",
        approval_probability=approval_probability,
        loan_to_income_ratio=loan_to_income_ratio,
        credit_score=credit_score,
        shap_top_drivers=shap_top_drivers,
        explanation_summary=explanation_summary,
    )
    return {
        "final_decision": "Refer",
        "triggered_rules": ["R3"],
        "decision_reason": decision_reason,
        "reviewer_required": True,
    }


def run(state: dict) -> dict:
    """LangGraph-node-style entry point."""
    feature_names = load_model_artifact()["feature_names"]
    return evaluate_policy(
        application=state["application"],
        feature_vector=state["feature_vector"],
        feature_names=feature_names,
        approval_probability=state["approval_probability"],
        shap_top_drivers=state.get("shap_top_drivers", []),
        explanation_summary=state.get("explanation_summary"),
    )
