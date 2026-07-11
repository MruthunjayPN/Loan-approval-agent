# CP07 — Build Log & Phase Tracker

Working log for this project. Updated at each iteration. Source-of-truth design docs live one level up (`../CP07-loan-approval-agent-brief.md`, `../phase0-task*.md`) — this file tracks *status*, not design.

**Ground rule in effect:** no generated code is accepted without the "why," judgment calls are flagged explicitly, understanding checkpoints happen before moving phases. See brief for full text.

---

## Phase status

- [x] **Phase 0 — Contracts & policy (done manually, pre-Claude Code)** — see the three phase0-task*.md docs. Complete.
- [x] **Phase 1 — Environment & Data** — code + training done, awaiting user's own explain-it-back checkpoint before Phase 2
  - [x] Repo scaffold
  - [x] Field-list discrepancy resolved (see Decisions log) — went with task3's fuller field list
  - [x] Data generation plan signed off by user
  - [x] `data/generate_data.py` written + explained
  - [x] Data generated (100,000 rows), spot-checked (aggregate correlations + individual high/low-risk profiles)
  - [x] `model/train_model.py` written + explained
  - [x] `model/model.pkl` trained (test accuracy 0.778, ROC-AUC 0.726)
  - [ ] Understanding checkpoint: predict_proba vs. label, XGBoost vs. logistic regression — explainable unprompted **(user to confirm before Phase 2 starts)**
- [ ] **Phase 2 — Agents 1 & 2 (Validation + Feature Engineering)** — not started, blocked on Phase 1 sign-off
- [ ] **Phase 3 — Agents 3 & 4 (ML Scoring + SHAP)** — not started
- [ ] **Phase 4 — Agent 5 (Policy + Decision) + graph wiring** — not started
- [ ] **Phase 5 — Gradio UI + edge cases** — not started
- [ ] **Phase 6 — README + explanation pass** — user writes this, not Claude Code

---

## Open questions / flagged judgment calls

*(none open right now)*

---

## Decisions log

- **Field list resolved (2026-07-11):** used task3's fuller field list — `age, annual_income, loan_amount, credit_score, employment_type, years_employed, existing_loans, default_history, label` — since task3's correlation #2 (years_employed → credit_score) needs a field the brief's shorthand list omitted. Brief's list treated as shorthand, not literal schema.
- **`loan_to_income_ratio` is never a stored column** in the raw data CSV — only computed internally (in generate_data.py to build labels, and again in train_model.py as a modeling feature). This preserves the AgentState boundary: Agent 2 (Feature Engineering) is supposed to be the one deriving it from raw fields later, not receiving it pre-computed.
- **`credit_score` is a raw input field**, not derived at feature-engineering time — its *value* is generated with correlations baked in (existing_loans, default_history, years_employed, employment_type all shift it), but it lives in the data like a real bureau score an applicant "brings with them," matching Agent 1's validation contract.
- **Risk-score weights, v1 → v2 recalibration:** initial weights (credit_score 0.35 / LTI 0.30 / default_history 0.20 / existing_loans 0.10 / unemployed 0.05) under-delivered on task3's "default_history → approval sharply down" requirement (only 72%→40% approval swing). Reweighted to 0.30 / 0.25 / 0.35 / 0.07 / 0.03 — now 78.5%→29%, a sharp drop, matching task3 and staying consistent with R1's hard-reject logic. This is the "fix the data, don't paper over it later" loop task3 asks for.
- **`employment_type` one-hot encoded with all 3 categories kept** (no dropped baseline) — XGBoost is tree-based and doesn't suffer from the multicollinearity a dropped-dummy avoids for linear models; keeping all 3 means every category shows up explicitly in SHAP later instead of one being an invisible reference category.
- **model.pkl stores `{model, feature_names, version}`**, not just the raw model — feature_names is the exact column order Agent 3 must reproduce at inference time, directly targeting the feature-ordering-mismatch failure mode task1 calls out for Agent 2.
- **Row count: 100,000**, per user request — functionally fine for both generation speed and XGBoost training time; only cost is CSV size (~5MB), acceptable for a static one-time artifact.
- **Data CSV and model.pkl ARE committed** (2026-07-11, final call, reversed twice in one session) — first defaulted to committing them (demo convenience), then switched to gitignoring them (prod practice), then switched back: this is a demo/portfolio repo meant to be cloned and run/shown immediately, and both files are reproducible via a fixed seed (42) regardless, so committing them costs nothing but convenience gained. Worth remembering for the README's "what's fake / what changes in production" section — real repos gitignore trained artifacts and pull them from a model registry instead.
- **libomp dependency:** XGBoost on macOS ARM needs the OpenMP runtime (`brew install libomp`) — not bundled with the pip wheel. Needed once per machine, not part of the Python venv.

---

## Iteration steps to follow each session

1. Re-read this file's Phase status + Open questions before generating anything new.
2. Don't start a phase until the prior phase's understanding checkpoint is explicitly confirmed by the user.
3. For any new script: explain the plan → get sign-off → write code → explain the code's key decisions inline → flag judgment calls here.
4. Update the Phase status checklist and Decisions log before ending the session.
