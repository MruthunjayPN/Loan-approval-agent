"""
graph.py — LangGraph wiring for the 5-agent loan approval pipeline.

Sequential: validation -> feature_engineering -> scoring -> shap -> policy.

Conditional edge after validation: FAILED short-circuits straight to a
terminal reject state, skipping feature_engineering/scoring/shap/policy
entirely — per Agent 1's contract in phase0-task1-agent-contracts.md
(don't spend a model or LLM call on invalid input).

No conditional edge after shap -> policy: shap_agent.run() already catches
every SHAP-computation failure internally (Phase 3) and always returns
normally, at worst with shap_top_drivers=[] / explanation_summary=None. A
plain sequential edge already satisfies "SHAP failure must not block
policy" — that guarantee lives inside the agent function, not the graph.
"""
from langgraph.graph import END, START, StateGraph

from agents.state import AgentState
from agents.validation_agent import run as run_validation
from agents.feature_engineering_agent import run as run_feature_engineering
from agents.scoring_agent import run as run_scoring
from agents.shap_agent import run as run_shap
from agents.policy_agent import run as run_policy


def invalid_input_reject(state: AgentState) -> dict:
    """Terminal node for FAILED validation. Deliberately not routed through
    policy_agent -- an invalid-input rejection isn't a policy rule firing,
    so it gets its own distinct triggered_rules marker (VALIDATION_FAILED)
    rather than being mislabeled as R1/R2/R3."""
    errors = state.get("validation_errors", [])
    return {
        "final_decision": "Rejected",
        "triggered_rules": ["VALIDATION_FAILED"],
        "decision_reason": f"Rejected before scoring: invalid input ({'; '.join(errors)}).",
        "reviewer_required": False,
    }


def route_after_validation(state: AgentState) -> str:
    return "feature_engineering" if state["validation_status"] == "PASSED" else "invalid_input_reject"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("validation", run_validation)
    builder.add_node("feature_engineering", run_feature_engineering)
    builder.add_node("scoring", run_scoring)
    builder.add_node("shap", run_shap)
    builder.add_node("policy", run_policy)
    builder.add_node("invalid_input_reject", invalid_input_reject)

    builder.add_edge(START, "validation")
    builder.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "feature_engineering": "feature_engineering",
            "invalid_input_reject": "invalid_input_reject",
        },
    )
    builder.add_edge("feature_engineering", "scoring")
    builder.add_edge("scoring", "shap")
    builder.add_edge("shap", "policy")  # plain edge -- SHAP degrades gracefully internally, not here
    builder.add_edge("policy", END)
    builder.add_edge("invalid_input_reject", END)

    return builder.compile()


GRAPH = build_graph()


def run_application(application: dict) -> AgentState:
    """Convenience entry point: run one raw application dict through the
    compiled graph end to end, returning the final AgentState."""
    return GRAPH.invoke({"application": application})
