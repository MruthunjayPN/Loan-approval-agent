# CP07 — Automated Loan Approval Agent

A 5-agent LangGraph pipeline that takes a raw loan application, validates it, scores it with a trained XGBoost model, explains the score with SHAP, and applies hardcoded policy rules — with an LLM invoked only to write the explanation for the ambiguous "Refer" cases, never to decide anything.

---

## Architecture

```
                    ┌────────────┐
   raw application  │            │
  ─────────────────▶│  Agent 1   │  Validation
                     │            │  required fields, types, bounds
                     └─────┬──────┘
                            │ validation_status
              ┌─────────────┴─────────────┐
              │ FAILED                     │ PASSED
              ▼                             ▼
     ┌──────────────────┐          ┌────────────┐
     │ invalid_input_    │          │  Agent 2   │  Feature Engineering
     │ reject (terminal) │          │            │  encode, derive loan_to_income_ratio,
     └──────────────────┘          └─────┬──────┘  order to match training
                                          │ feature_vector
                                          ▼
                                   ┌────────────┐
                                   │  Agent 3   │  ML Scoring
                                   │            │  model.pkl -> predict_proba
                                   └─────┬──────┘
                                          │ approval_probability
                                          ▼
                                   ┌────────────┐
                                   │  Agent 4   │  SHAP Explainability
                                   │            │  top 3 +/- drivers, plain-English summary
                                   └─────┬──────┘  (fails gracefully, never blocks)
                                          │ shap_top_drivers
                                          ▼
                                   ┌────────────┐
                                   │  Agent 5   │  Policy + Decision
                                   │            │  R1/R2/R3 hardcoded rules (see below)
                                   └─────┬──────┘  LLM only for Refer's decision_reason
                                          │
                                          ▼
                              final_decision / triggered_rules / decision_reason
```

One `AgentState` object flows through the graph. Each agent reads only what it needs and writes only the fields it owns — no agent mutates or deletes a prior agent's output. That makes every field traceable back to the single node that produced it.

### What each agent does — and doesn't do

- **Agent 1 — Validation**: required fields, types, and bounds (age 21–70, credit score range, positive income above a floor, valid enum values). On `FAILED`, a conditional edge short-circuits straight to a terminal reject node — feature engineering, scoring, SHAP, and the LLM are never invoked on garbage input. It does not touch policy or scoring logic.
- **Agent 2 — Feature Engineering**: derives `loan_to_income_ratio`, one-hot encodes `employment_type`, and orders the feature vector by reading `feature_names` directly out of `model.pkl` rather than hardcoding a second copy of the training-time column order. It never sees the raw payload directly — only a validated application — and it enforces `validation_status == "PASSED"` itself rather than trusting the caller.
- **Agent 3 — ML Scoring**: loads `model.pkl`, calls `predict_proba`, returns a probability plus a raw model-level `prediction_class` at a fixed 0.5 cutoff. That raw class is explicitly *not* the loan decision — only Agent 5's policy rules decide Approve/Reject/Refer. Retries once for transient model-load failures (file lock, cold start); a feature-vector length mismatch or a missing/corrupt file raises immediately, since that's a bug, not something a retry fixes.
- **Agent 4 — SHAP Explainability**: `TreeExplainer` computes per-prediction (local) attribution — top 3 positive and top 3 negative drivers plus a plain-English summary — as opposed to global feature importance, which only tells you what matters on average across the dataset, not for this applicant. SHAP failures are isolated to the SHAP computation itself and degrade gracefully: the decision proceeds without an explanation rather than blocking.
- **Agent 5 — Policy + Decision**: deterministic rules run first, in a fixed evaluation order, stop-on-first-match, implemented as plain Python early returns in one function (`evaluate_policy`) — not as graph-level branching. The LLM is invoked from *inside* that function, gated by which rule fired, never by a conditional edge in `graph.py`.

### The deterministic-vs-LLM boundary

R1, R2, and R3 are hardcoded and always decide the outcome before any LLM call happens:

1. **R1 — Hard reject floor**: `credit_score < 580 OR default_history == True → Reject`. Hardcoded because this is a compliance-adjacent floor — an LLM should never be able to talk its way around it.
2. **R2 — Affordability override**: `loan_to_income_ratio > 0.45 → Refer`, even if the model's own probability would say approve. This exists as a separate rule (not folded into R3's threshold) to demonstrate the policy layer overriding the model on a business signal the model wasn't necessarily weighting the same way.
3. **R3 — ML threshold triage** (only reached if R1/R2 didn't fire): `probability >= 0.75 → Approve`, `probability <= 0.40 → Reject`, otherwise `→ Refer`.

The LLM (Google Gemini) is called **0 or 1 times per request, never more** — only when R2 or R3's middle branch lands on Refer — and only to synthesize `decision_reason` from structured inputs already computed by prior agents (probability, SHAP drivers, near-miss margins to the R1/R2 boundaries). It cannot change Approve/Reject/Refer; that outcome is fixed in `evaluate_policy` before the LLM is ever called. If the LLM call fails for any reason, a deterministic templated reason is used instead — the same graceful-degradation principle Agent 4 uses for SHAP failures.

This gating lives entirely inside `policy_agent.py`'s business-rule branching, not as a conditional edge in `graph.py` — the graph itself is a plain sequential edge from `shap` to `policy` with no LLM-aware routing.

---

## Results

Trained on 100,000 synthetically generated rows, 11 engineered features, using an `XGBClassifier` (200 estimators, max depth 4, learning rate 0.1) with an 80/20 stratified train/test split (seed 42):

- **Test accuracy: 0.778**
- **Test ROC-AUC: 0.726**

These numbers come from re-running `model/train_model.py` against the committed data, not transcribed from an earlier log.

---

## Why this project

This is a portfolio project demonstrating multi-agent orchestration patterns with LangGraph — isolating responsibility across agents, giving each a narrow read/write contract over shared state, building a natural early-exit point after validation, and drawing a hard boundary between deterministic decision authority and LLM-assisted explanation. It is not a production credit model, and the policy thresholds and risk weighting are not researched or validated against real lending outcomes.

---

## What's real vs. what's fake here

- **Synthetic data.** All 100,000 applicant rows are generated (`data/generate_data.py`, fixed seed), not real applicant data.
- **Illustrative, not researched, policy thresholds.** The R1/R2/R3 numbers (580 credit score floor, 0.45 loan-to-income ceiling, 0.75/0.40 probability triage) are plausible defaults chosen to demonstrate the *pattern* of rule evaluation order, not numbers derived from actual credit risk research.
- **No real compliance or fair-lending review.** Nothing here has gone through a legal, regulatory, or fair-lending audit.
- **No drift monitoring.** Nothing tracks whether incoming applications or model performance are diverging from training-time assumptions over time.
- **No feature store.** Feature engineering happens inline in `agents/feature_engineering_agent.py`, not through a shared, versioned feature-serving system.
- **`model.pkl` is committed to git**, not pulled from a model registry. It's reproducible from a fixed seed either way, but a real deployment would not commit a trained artifact directly to source control.

---

## What I'd change for production

- **A real model registry** instead of a git-committed `model.pkl` — versioned, with lineage back to the training run and data snapshot that produced it.
- **Gitignored artifacts** (data and model) pulled from storage/registry at deploy or test time, not checked into the repo.
- **A feature store** so feature computation is shared, versioned, and consistent between training and inference instead of re-implemented in the agent.
- **Drift monitoring** — PSI or KS-statistic checks comparing live application/feature distributions against the training distribution, to catch silent model staleness.
- **A human review workflow for Refer outcomes** — right now a Refer just returns a `decision_reason`; there's no queue, assignment, or audit trail for the human reviewer it's meant to hand off to.
- **Out-of-distribution detection beyond validation bounds** — Agent 1's bounds checks catch clearly invalid input (matching the training data's sampling ranges), but nothing detects applications that are in-bounds field-by-field yet unlike anything the model was trained on.

---

## Setup / how to run

Requires Python 3.14 (tested with `python@3.14` via Homebrew). On macOS ARM, XGBoost also needs the OpenMP runtime, which isn't bundled with the pip wheel:

```bash
brew install libomp   # macOS ARM only, once per machine
```

Then, from the `cp07-loan-agent/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`data/loan_applicants.csv` and `model/model.pkl` are already committed and reproducible (fixed seed 42), so the app runs out of the box. To regenerate either from scratch:

```bash
python data/generate_data.py    # regenerates data/loan_applicants.csv
python model/train_model.py     # retrains and overwrites model/model.pkl, prints test accuracy/ROC-AUC
```

To use the LLM-synthesized `decision_reason` on Refer outcomes, create a `.env` file in `cp07-loan-agent/` with:

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.1-flash-lite
```

`GEMINI_API_KEY` is only required when a request actually lands on Refer (R2, or R3's middle branch) — Approve/Reject outcomes never touch the LLM client.

Then launch the UI:

```bash
python app.py
```

This starts a Gradio app (`gr.Blocks`) that takes a raw application through the full pipeline via `graph.run_application()` and displays the decision, reason, probability, triggered rule, and SHAP explanation.

---

## Interview talking points

- **Why 5 agents instead of one function?** Isolation of responsibility, independent testability per agent, a natural early-exit point after validation (no wasted model/LLM call on invalid input), and a clean, auditable boundary between deterministic and LLM-driven logic.
- **Why is R2 (affordability override) a separate rule from R3's probability threshold, instead of just another feature the model weighs?** It demonstrates the policy layer overriding the model's own probability based on a business signal — loan-to-income ratio — that the model wasn't necessarily weighting the way the business wants. R2 can force a Refer even when the model alone would say approve.
- **Why does Agent 2 read `feature_names` out of `model.pkl` instead of hardcoding the column order?** A second hardcoded copy of the training-time feature order would recreate the exact feature-ordering-mismatch failure mode this design is meant to prevent — single source of truth instead of two lists that can drift out of sync.
- **Why is the LLM call gated by policy logic inside `policy_agent.py`, not by graph-level routing in `graph.py`?** The Approve/Reject/Refer outcome is fully decided by `evaluate_policy`'s deterministic rules before the LLM is ever invoked — the LLM only writes the explanation text for an outcome that already exists, and can't influence it. Keeping that gate inside the function (not a conditional edge) keeps the graph itself dumb and sequential, and keeps decision authority entirely in one auditable place.
- **Why does the SHAP agent fail gracefully instead of raising?** Explainability failing shouldn't take down a lending decision — the try/except is scoped tightly around the SHAP computation itself (explainer, values, top-driver selection), while reading `feature_vector`/`approval_probability` off state happens outside that boundary, so a real wiring bug still raises instead of being silently swallowed as a "SHAP failure."
- **What did the two mislabeled test-fixture catches teach?** Twice during development, a test applicant was assumed to land in an "ambiguous" or "borderline" zone without actually running it through the pipeline first — one scored a confident 0.7978 approve (a moderate credit score doesn't stay borderline once `default_history=False` is factored in, since default history carries the largest risk weight), and one assumed to hit "R3 Approve" actually landed in R3 Refer (probability 0.7069, under the 0.75 bar — sitting exactly at the R1 credit-score floor isn't enough on its own to clear R3's separate approve threshold). Both were relabeled honestly to describe what the pipeline actually produces, rather than adjusted until they matched the original guess — and each mislabeling turned out to be a real finding about how the rules interact, not just a fixture bug.
