"""
Train XGBoost classifier on synthetic loan applicant data.

Saves model.pkl as {"model", "feature_names", "version"} — feature_names is
the exact column order Agent 3 must reproduce at inference time.
"""
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("data/loan_applicants.csv")

# Mirror what Agent 2 (Feature Engineering) will do later: derive the ratio,
# one-hot encode the categorical, coerce bool -> int.
df["loan_to_income_ratio"] = df["loan_amount"] / df["annual_income"]
df["default_history"] = df["default_history"].astype(int)
df = pd.get_dummies(df, columns=["employment_type"], prefix="employment")

feature_names = [c for c in df.columns if c != "label"]
X, y = df[feature_names], df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]  # P(label=1 / approved), not a label
preds = (probs >= 0.5).astype(int)  # threshold only for this eval metric, NOT the real decision logic
print(f"Test accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, probs):.3f}")

with open("model/model.pkl", "wb") as f:
    pickle.dump({"model": model, "feature_names": feature_names, "version": "v1"}, f)
print(f"Saved model/model.pkl ({len(feature_names)} features)")
