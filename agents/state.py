"""
Shared AgentState schema — the state object LangGraph threads through the
pipeline. Mirrors ../phase0-task1-agent-contracts.md field-for-field, plus
two additions logged in PROGRESS.md's decisions log: inference_latency
(Phase 3) and reviewer_required (Phase 4) aren't in the original contract
doc but were added for explicit, documented reasons.

TypedDict, not a dataclass: LangGraph's StateGraph requires a state_schema
type, and each agent only ever writes the handful of keys it owns
(append-only) — no single node populates every field at once, so
total=False.
"""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    application: dict[str, Any]

    validation_status: str  # "PASSED" | "FAILED"
    validation_errors: list[str]

    feature_vector: list[float]
    feature_version: str

    approval_probability: float
    prediction_class: str  # raw model-level label at a fixed cutoff -- NOT the policy decision
    model_version: str
    inference_latency: float

    shap_top_drivers: list[dict[str, Any]]
    explanation_summary: str | None

    final_decision: str  # "Approved" | "Rejected" | "Refer"
    triggered_rules: list[str]  # always length 1 in this design (stop-on-first-match)
    decision_reason: str
    reviewer_required: bool
