# CP07 — Build Log & Phase Tracker

Working log for this project. Updated at each iteration. Source-of-truth design docs live one level up (`../CP07-loan-approval-agent-brief.md`, `../phase0-task*.md`) — this file tracks *status*, not design.

**Ground rule in effect:** no generated code is accepted without the "why," judgment calls are flagged explicitly, understanding checkpoints happen before moving phases. See brief for full text.

---

## Phase status

- [ ] **Phase 0 — Contracts & policy (done manually, pre-Claude Code)** — see the three phase0-task*.md docs. Complete.
- [ ] **Phase 1 — Environment & Data** ← in progress
  - [ ] Repo scaffold
  - [ ] Field-list discrepancy resolved (see Open Questions)
  - [ ] Data generation plan signed off by user
  - [ ] `data/generate_data.py` written + explained
  - [ ] Data generated, spot-checked
  - [ ] `model/train_model.py` written + explained
  - [ ] `model/model.pkl` trained
  - [ ] Understanding checkpoint: predict_proba vs. label, XGBoost vs. logistic regression — explainable unprompted
- [ ] **Phase 2 — Agents 1 & 2 (Validation + Feature Engineering)** — not started, blocked on Phase 1 sign-off
- [ ] **Phase 3 — Agents 3 & 4 (ML Scoring + SHAP)** — not started
- [ ] **Phase 4 — Agent 5 (Policy + Decision) + graph wiring** — not started
- [ ] **Phase 5 — Gradio UI + edge cases** — not started
- [ ] **Phase 6 — README + explanation pass** — user writes this, not Claude Code

---

## Open questions / flagged judgment calls

1. **Field list mismatch (raised 2026-07-11, awaiting answer):** brief lists `age, income, loan_amount, credit_score, employment_type, existing_loans, default_history, label`; task3 requires `years_employed` (for correlation #2) and calls the income field `annual_income`. Defaulting to task3's fuller list as ground truth unless told otherwise.

---

## Decisions log

*(Append here as judgment calls get made and confirmed, so the "why" survives past this session.)*

---

## Iteration steps to follow each session

1. Re-read this file's Phase status + Open questions before generating anything new.
2. Don't start a phase until the prior phase's understanding checkpoint is explicitly confirmed by the user.
3. For any new script: explain the plan → get sign-off → write code → explain the code's key decisions inline → flag judgment calls here.
4. Update the Phase status checklist and Decisions log before ending the session.
