"""
BAN6800 Module 4 - Predictive Model Development & Validation Pack
Business need: Predict voluntary customer churn (yes/no) for a telecom
subscriber base so retention teams can intervene before contract renewal.
Dataset: IBM/Kaggle Telco Customer Churn (teleconnect.csv), 7,043 customers.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. LOAD DATA (local file only - no fabricated network fallback)
# ---------------------------------------------------------------
df = pd.read_csv("teleconnect.csv")
n_rows_raw = len(df)

# TotalCharges is stored as text with blank strings for 11 new customers
# (tenure = 0); coerce to numeric and drop the resulting nulls.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
n_missing_totalcharges = df["TotalCharges"].isna().sum()
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
n_rows_clean = len(df)

# Target encoding
y = df["Churn"].map({"Yes": 1, "No": 0})
churn_rate = y.mean()

# Drop identifier; keep customerID out of X to prevent leakage/no signal
X = df.drop(columns=["customerID", "Churn"])

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]

# ---------------------------------------------------------------
# 2. STRATIFIED TRAIN/TEST SPLIT (80/20) + 5-FOLD CV FOR TUNING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------
# 3. BASELINE (majority-class dummy classifier)
# ---------------------------------------------------------------
baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_preds = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, zero_division=0)

# ---------------------------------------------------------------
# 4. MODEL 1: LOGISTIC REGRESSION (+ grid search)
# ---------------------------------------------------------------
pipe_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(solver="liblinear", max_iter=1000,
                                        class_weight="balanced", random_state=RANDOM_STATE)),
])
grid_lr = GridSearchCV(
    pipe_lr,
    param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__penalty": ["l1", "l2"]},
    cv=5, scoring="f1", n_jobs=-1,
)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

# ---------------------------------------------------------------
# 5. MODEL 2: RANDOM FOREST (+ grid search)
# ---------------------------------------------------------------
pipe_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])
grid_rf = GridSearchCV(
    pipe_rf,
    param_grid={"classifier__n_estimators": [200, 400],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_leaf": [1, 5]},
    cv=5, scoring="f1", n_jobs=-1,
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# ---------------------------------------------------------------
# 6. TEST-SET EVALUATION
# ---------------------------------------------------------------
def evaluate(model, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "name": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "preds": preds,
        "probs": probs,
        "cm": confusion_matrix(y_test, preds),
        "report": classification_report(y_test, preds, target_names=["Retained", "Churned"]),
    }

res_lr = evaluate(best_lr, "Logistic Regression")
res_rf = evaluate(best_rf, "Random Forest")

# ---------------------------------------------------------------
# 7. FEATURE IMPORTANCE (Random Forest, mean decrease in impurity)
# ---------------------------------------------------------------
ohe = best_rf.named_steps["preprocessor"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_names
importances = best_rf.named_steps["classifier"].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
top_features = feat_imp.head(10)

# ---------------------------------------------------------------
# 8. FIGURES (4-panel validation pack)
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.heatmap(res_lr["cm"], annot=True, fmt="d", cmap="Blues", ax=axes[0, 0], cbar=False,
            xticklabels=["Pred: Retained", "Pred: Churned"], yticklabels=["True: Retained", "True: Churned"])
axes[0, 0].set_title("Figure 1. Logistic Regression Confusion Matrix")

sns.heatmap(res_rf["cm"], annot=True, fmt="d", cmap="Greens", ax=axes[0, 1], cbar=False,
            xticklabels=["Pred: Retained", "Pred: Churned"], yticklabels=["True: Retained", "True: Churned"])
axes[0, 1].set_title("Figure 2. Random Forest Confusion Matrix")

fpr_lr, tpr_lr, _ = roc_curve(y_test, res_lr["probs"])
fpr_rf, tpr_rf, _ = roc_curve(y_test, res_rf["probs"])
axes[1, 0].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={res_lr['roc_auc']:.3f})")
axes[1, 0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={res_rf['roc_auc']:.3f})")
axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC=0.500)")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_title("Figure 3. ROC Curves - Model Comparison")
axes[1, 0].legend(loc="lower right", fontsize=8)

top_features.sort_values().plot(kind="barh", ax=axes[1, 1], color="#2e7d32")
axes[1, 1].set_title("Figure 4. Random Forest Top 10 Feature Importances")
axes[1, 1].set_xlabel("Mean Decrease in Impurity")

plt.tight_layout()
plt.savefig("validation_artifacts.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 9. SAVE ALL NUMBERS TO JSON FOR THE REPORT (no hand-typed stats)
# ---------------------------------------------------------------
summary = {
    "n_rows_raw": int(n_rows_raw),
    "n_missing_totalcharges": int(n_missing_totalcharges),
    "n_rows_clean": int(n_rows_clean),
    "churn_rate": round(float(churn_rate), 4),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "baseline_accuracy": round(float(baseline_acc), 4),
    "baseline_f1": round(float(baseline_f1), 4),
    "lr_best_params": grid_lr.best_params_,
    "rf_best_params": grid_rf.best_params_,
    "lr": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res_lr.items() if k in
           ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    "rf": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res_rf.items() if k in
           ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    "lr_cm": res_lr["cm"].tolist(),
    "rf_cm": res_rf["cm"].tolist(),
    "top_features": {k: round(float(v), 4) for k, v in top_features.items()},
}

with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\n--- LR classification report ---")
print(res_lr["report"])
print("\n--- RF classification report ---")
print(res_rf["report"])
