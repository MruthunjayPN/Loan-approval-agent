"""
Agent 2 — Feature Engineering.

Reads a validated application and writes feature_vector / feature_version
onto AgentState. See ../phase0-task1-agent-contracts.md for the full
contract.

The feature order is never hardcoded here as a second list — it's read
directly from model.pkl['feature_names'], the single source of truth
PROGRESS.md's decisions log designated for training-time column order.
Duplicating that order in this file would recreate the exact
feature-ordering-mismatch failure mode phase0-task1 warns about for Agent 2.
"""
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

# Resolved relative to this file, not cwd — agents get invoked from graph.py,
# tests, and (later) Gradio, which won't all share one working directory.
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.pkl"

ALLOWED_EMPLOYMENT_TYPES = {"salaried", "self_employed", "unemployed"}


@lru_cache(maxsize=1)
def load_model_artifact() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def engineer_features(application: dict[str, Any]) -> tuple[list[float], str]:
    """Pure function: validated application dict in, (feature_vector,
    feature_version) out."""
    employment_type = application["employment_type"]
    if employment_type not in ALLOWED_EMPLOYMENT_TYPES:
        # Validation should already have caught this. Guarding again here
        # matters because the one-hot flags below are three fixed named
        # keys (not built dynamically from the input value) — an unknown
        # category would otherwise silently resolve all three to 0 instead
        # of raising, producing a *valid-shaped* but wrong feature vector.
        # That's the "malformed feature vector" failure mode called out in
        # the brief, and it would not be caught by the missing/extra check
        # below since no extra key is ever generated.
        raise ValueError(
            f"unknown employment_type {employment_type!r} reached feature "
            "engineering — validation should have rejected this upstream"
        )

    computed: dict[str, Any] = {
        "age": application["age"],
        "annual_income": application["annual_income"],
        "loan_amount": application["loan_amount"],
        "credit_score": application["credit_score"],
        "years_employed": application["years_employed"],
        "existing_loans": application["existing_loans"],
        "default_history": int(application["default_history"]),
        "loan_to_income_ratio": application["loan_amount"] / application["annual_income"],
        "employment_salaried": int(employment_type == "salaried"),
        "employment_self_employed": int(employment_type == "self_employed"),
        "employment_unemployed": int(employment_type == "unemployed"),
    }

    artifact = load_model_artifact()
    feature_names = artifact["feature_names"]

    # Defense in depth on top of the explicit employment_type check above:
    # catches drift the other direction too, e.g. train_model.py adding a
    # new engineered feature this agent was never updated to produce.
    missing = [name for name in feature_names if name not in computed]
    extra = [name for name in computed if name not in feature_names]
    if missing or extra:
        raise ValueError(
            f"feature engineering / model.pkl mismatch — missing: {missing}, extra: {extra}"
        )

    feature_vector = [computed[name] for name in feature_names]
    return feature_vector, artifact["version"]


def run(state: dict) -> dict:
    """LangGraph-node-style entry point. Enforces the PASSED-only
    precondition itself rather than trusting graph.py to route correctly —
    the check is cheap and the failure mode it prevents (engineering
    features for an invalid application) is worse than the cost of it."""
    if state.get("validation_status") != "PASSED":
        raise RuntimeError(
            "feature_engineering_agent called on a non-PASSED application "
            f"(validation_status={state.get('validation_status')!r}) — "
            "graph.py should route FAILED applications straight to rejection"
        )

    feature_vector, feature_version = engineer_features(state["application"])
    return {"feature_vector": feature_vector, "feature_version": feature_version}
