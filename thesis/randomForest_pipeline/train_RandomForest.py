# Random Forest pipeline using thesis utility modules

import sys
import os
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ── path setup ──────────────────────────────────────────────────────────────
# This script lives at thesis/randomForest_pipeline/train_RandomForest.py
# The shared utilities live one level up (thesis/)
SCRIPT_DIR  = Path(__file__).resolve().parent          # .../thesis/randomForest_pipeline
THESIS_DIR  = SCRIPT_DIR.parent                        # .../thesis
CODE_DIR    = THESIS_DIR.parent                        # .../Code

sys.path.insert(0, str(THESIS_DIR))   # import load_data, feature_engineering, etc.

from load_data         import load_data
from feature_engineering import engineer_features
from split_data        import split_data
from selection         import select_features


# ── helpers ─────────────────────────────────────────────────────────────────
def train_rf(X_train, y_train,
             n_estimators=100, max_depth=20,
             min_samples_split=7, random_state=42):
    """Train and return a RandomForestClassifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate(model, X_train, X_test, y_train, y_test,
             threshold=0.3, predict_proba=True):
    """Print and return a dict of evaluation metrics."""
    if predict_proba and hasattr(model, 'predict_proba'):
        y_prob  = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_prob >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_pred  = model.predict(X_test)
        roc_auc = None

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc,
    }

    print("=" * 40)
    print("Test-set evaluation")
    print("=" * 40)
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:12s}: {v:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


# ── main pipeline ────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore")

    # 1. Load raw data
    print("Loading data …")
    data = load_data(processed=False)

    # 2. Feature engineering (diag grouping, medication encoding,
    #    A1Cresult, label encoding, binary target)
    print("Engineering features …")
    data = engineer_features(data, target="readmitted")

    #drop additional column 
    data.drop("encounter_id", axis=1, inplace=True)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = split_data(
        data, target="readmitted", test_size=0.2, random_state=42
    )
    print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    # 4. (Optional) feature selection – uncomment to enable
    print('\nFeatures before selection ({}):'.format(X_train.shape[1]))
    print(list(X_train.columns))
    X_train, X_test = select_features(
        "RandomForest", X_train, X_test, y_train, step=1, cv=5
    )
    print('\nFeatures after selection ({}):'.format(X_train.shape[1]))
    print(list(X_train.columns))

    # 5. Train
    print("Training Random Forest …")
    rf = train_rf(X_train, y_train,
                  n_estimators=100, max_depth=20,
                  min_samples_split=7, random_state=42)

    # 6. Evaluate
    metrics = evaluate(rf, X_train, X_test, y_train, y_test, threshold=0.3)

    # 7. MLflow logging
    mlflow_dir = THESIS_DIR / "experiments" / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("file://" + str(mlflow_dir))
    mlflow.set_experiment("random_forest_pipeline")

    with mlflow.start_run():
        # parameters
        mlflow.log_params({
            "n_estimators":      rf.get_params()["n_estimators"],
            "max_depth":         rf.get_params()["max_depth"],
            "min_samples_split": rf.get_params()["min_samples_split"],
            "min_samples_leaf":  rf.get_params()["min_samples_leaf"],
            "criterion":         rf.get_params()["criterion"],
            "n_features":        X_train.shape[1],
            "test_size":         0.2,
            "threshold":         0.3,
        })
        # metrics
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        # model artifact with explicit name/signature/input_example
        input_example = X_train.head(5)
        signature = infer_signature(X_train, rf.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=rf,
            name="random_forest_model",
            signature=signature,
            input_example=input_example,
        )

    print("\nMLflow run logged to:", mlflow_dir)


if __name__ == "__main__":
    main()
