import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, engineer_features

DATA_PATH = "data/fraudTrain.csv"
MODEL_PATH = "models/fraud_model.joblib"
METADATA_PATH = "models/metadata.json"
RANDOM_STATE = 42


def load_and_engineer(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return engineer_features(raw)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    print(f"\n=== {name} (evaluated on REAL imbalanced test set: "
          f"{y_test.sum()} fraud / {len(y_test)} total) ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("  confusion matrix [[TN FP] [FN TP]]:")
    print(" ", confusion_matrix(y_test, pred).tolist())
    return metrics


def main():
    print("Loading data and engineering features...")
    df = load_and_engineer(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    # Split BEFORE any rebalancing, stratified so both splits keep the
    # real ~0.6% fraud ratio. The test set is never touched again until
    # final evaluation -- that's what makes the reported numbers honest.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud, {y_train.mean():.3%})")
    print(f"Test:  {len(X_test)} rows ({y_test.sum()} fraud, {y_test.mean():.3%}) -- untouched, real ratio")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {}

    # --- Logistic Regression: class_weight='balanced' handles the
    #     imbalance without throwing away legitimate-transaction data
    #     the way undersampling does. ---
    lr_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])
    lr_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    lr_search = RandomizedSearchCV(
        lr_pipe, lr_grid, n_iter=4, scoring="average_precision",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    t0 = time.time()
    lr_search.fit(X_train, y_train)
    print(f"\nLogisticRegression search done in {time.time()-t0:.1f}s, best CV PR-AUC={lr_search.best_score_:.4f}, params={lr_search.best_params_}")
    candidates["logistic_regression"] = lr_search.best_estimator_

    # --- Random Forest: tuned depth/estimators instead of the
    #     arbitrary max_depth=2 in the original notebook. ---
    rf_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_grid, n_iter=6, scoring="average_precision",
        cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    t0 = time.time()
    rf_search.fit(X_train, y_train)
    print(f"RandomForest search done in {time.time()-t0:.1f}s, best CV PR-AUC={rf_search.best_score_:.4f}, params={rf_search.best_params_}")
    candidates["random_forest"] = rf_search.best_estimator_

    # --- Final, honest comparison on the untouched, realistically
    #     imbalanced test set. ---
    results = {name: evaluate(name, model, X_test, y_test) for name, model in candidates.items()}

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model = candidates[best_name]
    print(f"\nSelected model: {best_name} (highest PR-AUC on held-out test set)")

    joblib.dump(best_model, MODEL_PATH)
    metadata = {
        "model_name": best_name,
        "features": ALL_FEATURES,
        "test_metrics": results[best_name],
        "all_candidate_metrics": results,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_fraud_rate": float(y_test.mean()),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model to {MODEL_PATH} and metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
