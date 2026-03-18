# XGBoost pipeline using thesis utility modules

import argparse
import sys
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from mlflow.models import infer_signature
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# -- path setup --------------------------------------------------------------
# This script lives at thesis/xgboost_pipeline/train_XGBoost.py
# The shared utilities live one level up (thesis/)
SCRIPT_DIR = Path(__file__).resolve().parent          # .../thesis/xgboost_pipeline
THESIS_DIR = SCRIPT_DIR.parent                        # .../thesis

sys.path.insert(0, str(THESIS_DIR))  # import selection utility

from selection import select_features


# -- helpers ----------------------------------------------------------------
def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train XGBoost with explicit train/test CSV files.")
    parser.add_argument(
        "--data-source",
        choices=["default", "pategan", "healthgan"],
        default="default",
        help="Preset source for train/test CSV paths.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Optional override for training CSV path.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Optional override for test CSV path.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="readmitted",
        help="Target column name.",
    )
    return parser.parse_args()


def resolve_data_paths(data_source):
    """Resolve default train/test CSV paths for a given source."""
    data_dir = THESIS_DIR / "data"
    default_train = data_dir / "diabetic_data_preprocessed_train.csv"
    default_test = data_dir / "diabetic_data_preprocessed_test.csv"

    if data_source == "default":
        return default_train, default_test

    if data_source == "pategan":
        return data_dir / "pategan" / "diabetic_data_pategan_train_synthetic_epsilon_5.0.csv", default_test

    # healthgan: use first synthetic CSV found in thesis/data/healthgan
    healthgan_dir = data_dir / "healthgan"
    healthgan_candidates = sorted(healthgan_dir.glob("*synthetic*.csv"))
    if not healthgan_candidates:
        raise FileNotFoundError(f"No HealthGAN synthetic CSV found in: {healthgan_dir}")
    return healthgan_candidates[0], default_test


def train_xgb(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
):
    """Train and return an XGBClassifier."""
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    return xgb


def evaluate(model, X_train, X_test, y_train, y_test, threshold=0.3, predict_proba=True):
    """Print and return a dict of evaluation metrics."""
    if predict_proba and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_pred = model.predict(X_test)
        roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
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


# -- main pipeline -----------------------------------------------------------
def main(args):
    warnings.filterwarnings("ignore")

    # 1. Resolve train/test CSV paths
    default_train_csv, default_test_csv = resolve_data_paths(args.data_source)
    train_csv = Path(args.train_csv) if args.train_csv else default_train_csv
    test_csv = Path(args.test_csv) if args.test_csv else default_test_csv

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # 2. Load train/test datasets
    print(f"Loading train data from: {train_csv}")
    train_data = pd.read_csv(train_csv)
    print(f"Loading test data from: {test_csv}")
    test_data = pd.read_csv(test_csv)

    if args.target not in train_data.columns:
        raise ValueError(f"Target column '{args.target}' not found in train CSV.")
    if args.target not in test_data.columns:
        raise ValueError(f"Target column '{args.target}' not found in test CSV.")

    # 3. Build X/y from separate datasets
    X_train = train_data.drop(columns=[args.target])
    y_train = train_data[args.target]
    X_test = test_data.drop(columns=[args.target])
    y_test = test_data[args.target]
    print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    # 4. Feature selection
    print('\nFeatures before selection ({}):'.format(X_train.shape[1]))
    print(list(X_train.columns))
    X_train, X_test = select_features(
        "XGBoost", X_train, X_test, y_train, step=1, cv=5
    )
    print('\nFeatures after selection ({}):'.format(X_train.shape[1]))
    print(list(X_train.columns))
    # 5. Train
    print("Training XGBoost ...")
    xgb = train_xgb(
        X_train,
        y_train,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    # 6. Evaluate on provided test CSV
    metrics = evaluate(xgb, X_train, X_test, y_train, y_test, threshold=0.3)

    # 7. MLflow logging
    mlflow_dir = THESIS_DIR / "experiments" / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("file://" + str(mlflow_dir))
    mlflow.set_experiment("xgboost_pipeline")

    with mlflow.start_run():
        # parameters
        mlflow.log_params({
            "n_estimators": xgb.get_params()["n_estimators"],
            "max_depth": xgb.get_params()["max_depth"],
            "learning_rate": xgb.get_params()["learning_rate"],
            "subsample": xgb.get_params()["subsample"],
            "colsample_bytree": xgb.get_params()["colsample_bytree"],
            "objective": xgb.get_params()["objective"],
            "n_features": X_train.shape[1],
            "threshold": 0.3,
            "data_source": args.data_source,
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "target": args.target,
        })
        # metrics
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        # model artifact
        input_example = X_train.head(5)
        signature = infer_signature(X_train, xgb.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=xgb,
            name="xgboost_model",
            signature=signature,
            input_example=input_example,
        )

    print("\nMLflow run logged to:", mlflow_dir)


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
