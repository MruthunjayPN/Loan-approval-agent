"""
Agent 3 — ML Scoring.

Reads feature_vector/feature_version from state only — never reaches into
state['application'] directly, since Agent 2 already owns turning raw
input into features. Writes approval_probability, prediction_class,
model_version, inference_latency. See ../phase0-task1-agent-contracts.md
for the Agent 3 contract this implements.

predict_proba returns a probability, not a label. prediction_class below
is a raw model-level label at a fixed 0.5 cutoff, for logging/metrics
only — mirrors train_model.py's eval-only cutoff. It is NOT the lending
decision; Agent 5's policy rules own that.
"""
import logging
import pickle
import time
from typing import Any

from agents.feature_engineering_agent import load_model_artifact

logger = logging.getLogger(__name__)

PREDICTION_CLASS_THRESHOLD = 0.5  # eval/logging only — NOT the policy decision boundary


def _load_model_with_retry() -> dict:
    """Loads the model artifact, retrying once only on a transient load
    failure (file lock / cold-start race). Never retries a missing or
    corrupt file — that's a bug, not a transient condition, and retrying
    would just hide it behind a delay."""
    try:
        return load_model_artifact()
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(f"model artifact missing or corrupt: {e}") from e
    except OSError as e:
        logger.warning("transient model-load failure (%s) — retrying once", e)
        try:
            return load_model_artifact()
        except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e2:
            raise RuntimeError(f"model artifact missing or corrupt on retry: {e2}") from e2
        except OSError as e2:
            raise RuntimeError(
                f"model artifact still failing to load after one retry: {e2}"
            ) from e2


def score_application(feature_vector: list[float], feature_version: str) -> dict[str, Any]:
    """feature_vector/feature_version in, scoring fields out. Only side
    effect is reading model.pkl via the shared cached loader."""
    artifact = _load_model_with_retry()
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_version = artifact["version"]

    # Loud and immediate, no retry — a length mismatch means Agent 2 drifted
    # from model.pkl's contract or feature_version is stale, not a fluke.
    if len(feature_vector) != len(feature_names):
        raise ValueError(
            f"feature_vector length {len(feature_vector)} does not match "
            f"model's expected {len(feature_names)} features "
            f"(feature_version={feature_version!r}, model_version={model_version!r})"
        )

    start = time.perf_counter()
    approval_probability = float(model.predict_proba([feature_vector])[0, 1])
    inference_latency = time.perf_counter() - start

    prediction_class = (
        "approve" if approval_probability >= PREDICTION_CLASS_THRESHOLD else "reject"
    )

    return {
        "approval_probability": approval_probability,
        "prediction_class": prediction_class,
        "model_version": model_version,
        "inference_latency": inference_latency,
    }


def run(state: dict) -> dict:
    """LangGraph-node-style entry point."""
    return score_application(state["feature_vector"], state["feature_version"])
