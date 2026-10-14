"""Model loading and single-instance prediction entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from xgboost import XGBClassifier

from asoc_ml.config import CLASS_NAMES, MODELS_DIR

DEFAULT_FEATURES = [
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


def load_model(model_path: str | None = None) -> XGBClassifier:
    """Load the trained native XGBoost model from disk."""
    load_dotenv()
    resolved_path = Path(
        model_path or os.getenv("MODEL_PATH", str(MODELS_DIR / "triage_model.json"))
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found at {resolved_path}.")

    model = XGBClassifier()
    model.load_model(str(resolved_path))
    return model


def _load_metadata() -> dict[str, object]:
    metadata_path = MODELS_DIR / "metadata.json"
    if not metadata_path.exists():
        return {"feature_names": DEFAULT_FEATURES, "class_names": CLASS_NAMES}
    with metadata_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def predict_one(features: dict) -> tuple[str, dict[str, float], float, list[dict[str, float]]]:
    """Return a verdict, probability distribution, confidence, and top features.

    TODO: replace the current feature-importance-based explanation with SHAP values
    once the model is stable and explainability is required by the product team.
    """
    model = load_model()
    metadata = _load_metadata()
    feature_names = list(metadata.get("feature_names", DEFAULT_FEATURES))
    class_names = list(metadata.get("class_names", CLASS_NAMES))

    vector = [float(features.get(name, 0.0)) for name in feature_names]
    probabilities = model.predict_proba(np.asarray([vector], dtype=float))[0]
    predicted_index = int(np.argmax(probabilities))
    verdict = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    probability_map = {
        class_name: float(probability)
        for class_name, probability in zip(class_names, probabilities, strict=False)
    }

    importances = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    top_features = [
        {"name": name, "impact": float(importance)}
        for name, importance in ranked[:3]
    ]

    return verdict, probability_map, confidence, top_features
