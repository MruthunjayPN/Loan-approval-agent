"""
Agent 4 — SHAP Explainability.

Reads model, feature_vector, approval_probability. Writes
shap_top_drivers (top 3 positive + top 3 negative contributors) and
explanation_summary onto state. See ../phase0-task1-agent-contracts.md
for the Agent 4 contract this implements.

If SHAP computation fails, this agent degrades gracefully rather than
blocking the pipeline: it logs a warning and writes an empty driver list
plus a null summary, letting Agent 5 proceed on probability + policy
rules alone. The try/except boundary sits only around the actual SHAP
computation (explainer + shap_values + top-driver selection) — reading
feature_vector/approval_probability off state happens outside it, in
run(), since a missing key there means the graph is wired wrong, not
that SHAP failed.

Contribution magnitudes are in log-odds (margin) space, not probability
points — verified during development by checking that
sum(shap_values) + expected_value reproduces the log-odds of
approval_probability, not the probability itself, for TreeExplainer on
this XGBClassifier. Direction (push toward approve vs. reject) is valid;
the number is not "probability points."
"""
import logging
from typing import Any

import numpy as np
import shap

from agents.feature_engineering_agent import load_model_artifact

logger = logging.getLogger(__name__)

TOP_N = 3


def _build_top_drivers(
    shap_values: np.ndarray, feature_names: list[str]
) -> tuple[list[dict[str, Any]], str]:
    pairs = list(zip(feature_names, shap_values.tolist()))

    positive = sorted((p for p in pairs if p[1] > 0), key=lambda p: p[1], reverse=True)[:TOP_N]
    negative = sorted((p for p in pairs if p[1] < 0), key=lambda p: p[1])[:TOP_N]

    drivers = [
        {"feature": name, "contribution": value, "direction": "approve"}
        for name, value in positive
    ] + [
        {"feature": name, "contribution": value, "direction": "reject"}
        for name, value in negative
    ]

    top_positive = positive[0][0] if positive else None
    top_negative = negative[0][0] if negative else None
    if top_positive and top_negative:
        summary = (
            f"{top_positive} pushed this prediction toward approval most strongly, "
            f"while {top_negative} pushed it most strongly toward rejection."
        )
    elif top_positive:
        summary = f"{top_positive} was the strongest driver, pushing toward approval."
    elif top_negative:
        summary = f"{top_negative} was the strongest driver, pushing toward rejection."
    else:
        summary = "No feature had a meaningful effect on this prediction."

    return drivers, summary


def explain_prediction(
    feature_vector: list[float], approval_probability: float
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        artifact = load_model_artifact()
        model = artifact["model"]
        feature_names = artifact["feature_names"]

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(np.array([feature_vector]))
        shap_values = np.asarray(raw_values)[0]

        # Judgment call: actively use approval_probability (which the
        # contract lists as an Agent 4 input) as a cross-check rather than
        # reading and discarding it. Converts the log-odds SHAP values back
        # through a sigmoid and compares to Agent 3's probability — a
        # meaningful divergence would mean Agent 3 and Agent 4 disagree
        # about the same prediction, which is worth knowing about.
        expected_value = explainer.expected_value
        if hasattr(expected_value, "__len__"):
            expected_value = expected_value[0]
        implied_probability = 1 / (1 + np.exp(-(shap_values.sum() + expected_value)))
        if abs(implied_probability - approval_probability) > 0.01:
            logger.warning(
                "SHAP-implied probability (%.4f) diverges from Agent 3's "
                "approval_probability (%.4f) by more than tolerance",
                implied_probability,
                approval_probability,
            )

        return _build_top_drivers(shap_values, feature_names)
    except Exception as e:
        logger.warning("SHAP computation failed, degrading gracefully: %s", e)
        return [], None


def run(state: dict) -> dict:
    """LangGraph-node-style entry point. feature_vector and
    approval_probability are read here, outside explain_prediction's
    try/except — a missing key means the graph is wired wrong, which must
    raise, not be swallowed as a SHAP failure."""
    feature_vector = state["feature_vector"]
    approval_probability = state["approval_probability"]

    shap_top_drivers, explanation_summary = explain_prediction(
        feature_vector, approval_probability
    )
    return {"shap_top_drivers": shap_top_drivers, "explanation_summary": explanation_summary}
