"""
Synthetic loan applicant data generator.

Produces RAW applicant fields only (mirrors AgentState.application) — derived
features like loan_to_income_ratio are computed internally here just to build
the label, but are NOT stored as a column. That's Agent 2's job later; the
CSV should look like data a real applicant would submit, not pre-engineered
features.
"""
import numpy as np
import pandas as pd

N_ROWS = 100_000
SEED = 42


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def generate_data(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 71, size=n)

    employment_type = rng.choice(
        ["salaried", "self_employed", "unemployed"], size=n, p=[0.65, 0.31, 0.04]
    )

    # years_employed bounded by age (can't exceed working years); unemployed
    # applicants get a short "current stint" range instead of a long tenure.
    max_possible = np.clip(age - 18, 1, None)
    years_employed = rng.integers(0, max_possible + 1)
    years_employed = np.where(
        employment_type == "unemployed", rng.integers(0, 2, size=n), years_employed
    )

    existing_loans = rng.integers(0, 5, size=n)
    default_history = rng.random(n) < 0.12

    # credit_score is a raw input field (like a real bureau score) but its
    # VALUE is generated with the task3 correlations baked in.
    credit_score = rng.normal(680, 60, size=n)
    credit_score -= existing_loans * 12
    credit_score -= default_history * 90
    credit_score += years_employed * 2
    credit_score -= (employment_type == "unemployed") * 40
    credit_score = np.clip(credit_score, 300, 850).round().astype(int)

    # annual_income: lognormal (right-skewed, like real income), scaled by
    # employment type.
    base_income = rng.lognormal(mean=10.9, sigma=0.45, size=n)
    income_multiplier = np.select(
        [employment_type == "unemployed", employment_type == "self_employed"],
        [0.35, 1.05],
        default=1.0,
    )
    annual_income = np.clip(base_income * income_multiplier, 8000, None).round(2)

    # loan_amount scaled relative to income so loan_to_income_ratio has a
    # realistic, non-degenerate spread instead of being independently random.
    loan_amount = (annual_income * rng.uniform(0.05, 0.9, size=n)).round(2)
    loan_to_income_ratio = loan_amount / annual_income  # used below only

    # --- weighted risk score -> probability -> label (task3's approach) ---
    risk = (
        0.30 * normalize(580 - credit_score)
        + 0.25 * normalize(loan_to_income_ratio)
        + 0.35 * default_history.astype(float)
        + 0.07 * normalize(existing_loans)
        + 0.03 * (employment_type == "unemployed").astype(float)
    )
    noise = rng.normal(0, 0.08, size=n)
    approval_probability = 1 / (1 + np.exp((risk + noise - 0.5) * 6))
    label = (rng.random(n) < approval_probability).astype(int)  # 1=approved, 0=rejected

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "employment_type": employment_type,
            "years_employed": years_employed,
            "existing_loans": existing_loans,
            "default_history": default_history,
            "label": label,
        }
    )


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/loan_applicants.csv", index=False)

    print(f"Generated {len(df)} rows -> data/loan_applicants.csv")
    print("\nLabel balance:")
    print(df["label"].value_counts(normalize=True).round(3))
    print("\nMean feature values by label (sanity check — should differ):")
    print(
        df.assign(loan_to_income_ratio=df["loan_amount"] / df["annual_income"])
        .groupby("label")[["credit_score", "loan_to_income_ratio", "existing_loans"]]
        .mean()
        .round(3)
    )
