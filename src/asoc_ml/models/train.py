"""Train and save baseline models for incident triage."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from asoc_ml.config import CLASS_NAMES, LABEL_COL, MODELS_DIR, PROCESSED_DIR
from asoc_ml.data.load import load_data, make_synthetic_incidents
from asoc_ml.data.splits import make_splits
from asoc_ml.features.build import build_features
from asoc_ml.models.evaluate import evaluate

FEATURE_COLUMNS = [
    "org_id",
    "alert_count",
    "distinct_mitre_techniques",
    "distinct_entity_types",
    "distinct_categories",
    "hour_of_day",
    "n_devices",
    "n_accounts",
    "n_ips",
]


def _prepare_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load processed features if present; otherwise generate synthetic data."""
    processed_path = PROCESSED_DIR / "incident_features.parquet"
    if processed_path.exists():
        print(f"Loading processed features from {processed_path}")
        df = load_data(processed_path)
    else:
        print("Warning: no processed parquet found. Falling back to synthetic incident data.")
        synthetic_df = make_synthetic_incidents(n=2000, seed=42)
        df = build_features(synthetic_df)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(processed_path)

    train_df, val_df, test_df = make_splits(
        df,
        "Timestamp",
        "incident_id",
        train_frac=0.7,
        val_frac=0.15,
    )
    return train_df, val_df, test_df


def _create_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(df.select(FEATURE_COLUMNS).to_numpy(), dtype=float)
    y_values = df.get_column(LABEL_COL).to_numpy()
    return x_values, y_values


def _encode_target(y: np.ndarray) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(CLASS_NAMES)}
    return np.asarray([mapping[label] for label in y], dtype=int)


def _decode_target(y_encoded: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_NAMES[int(value)] for value in y_encoded], dtype=object)


def main() -> None:
    train_df, val_df, _ = _prepare_data()
    x_train, y_train = _create_xy(train_df)
    x_val, y_val = _create_xy(val_df)

    y_train_encoded = _encode_target(y_train)
    _ = _encode_target(y_val)

    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        ),
    }

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_encoded)
    models["XGBClassifier"] = XGBClassifier(
        objective="multi:softprob",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    results: dict[str, float] = {}
    best_name = ""
    best_model = None
    best_macro_f1 = -1.0

    for name, model in models.items():
        if name == "XGBClassifier":
            model.fit(x_train, y_train_encoded, sample_weight=sample_weights)
            predictions = model.predict(x_val)
            predictions_decoded = _decode_target(predictions)
        else:
            model.fit(x_train, y_train)
            predictions = model.predict(x_val)
            predictions_decoded = predictions
        metrics = evaluate(y_val, predictions_decoded, CLASS_NAMES)
        results[name] = float(metrics["macro_f1"])

        if results[name] > best_macro_f1:
            best_macro_f1 = results[name]
            best_name = name
            best_model = model

    print("\nModel comparison (macro-F1):")
    for name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {score:.4f}")

    if best_model is None or best_name == "":
        raise RuntimeError("No model was trained successfully.")

    model_path = MODELS_DIR / "triage_model.json"
    METADATA_PATH = MODELS_DIR / "metadata.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    xgb_model = models["XGBClassifier"]
    xgb_model.save_model(str(model_path))

    metadata = {
        "model_version": "0.1.0",
        "feature_names": FEATURE_COLUMNS,
        "class_names": CLASS_NAMES,
        "macro_f1": best_macro_f1,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best model to {model_path}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
