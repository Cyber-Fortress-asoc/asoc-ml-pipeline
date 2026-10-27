import json
import logging
from datetime import datetime
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from asoc_ml.config import MODELS_DIR, DATA_DIR, LABEL_COL, CLASS_NAMES, TIME_COL, ID_COL
from asoc_ml.data.load import make_synthetic_incidents
from asoc_ml.data.splits import make_splits
from asoc_ml.features.build import build_features
from asoc_ml.models.evaluate import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    processed_path = DATA_DIR / "processed" / "incident_features.parquet"
    
    # 1. LOAD DATA: Check for real data, fallback to synthetic in-memory
    if processed_path.exists():
        logger.info(f"Loading REAL processed features from {processed_path}")
        df = pl.read_parquet(processed_path)
        data_source = "real"
    else:
        logger.warning("REAL DATA NOT FOUND! Generating in-memory synthetic features.")
        logger.warning("These will NOT be saved to data/processed/ to prevent data poisoning.")
        raw_synthetic = make_synthetic_incidents(n=2000, seed=42)
        df = build_features(raw_synthetic)
        data_source = "synthetic"

    # 2. SPLIT DATA
    logger.info("Splitting data (Time-based, grouped by incident)...")
    train_df, val_df, test_df = make_splits(df, time_col="incident_start_time", group_col=ID_COL)
    
    # Extract feature columns (drop ID, Time, and Label columns)
    drop_cols = [ID_COL, "org_id", "incident_start_time", LABEL_COL]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    # Convert Polars to Pandas for Scikit-Learn/XGBoost compatibility
    X_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df[LABEL_COL].to_pandas()
    X_val = val_df.select(feature_cols).to_pandas()
    y_val = val_df[LABEL_COL].to_pandas()

    # 3. TRAIN BASELINES
    logger.info("Training Dummy Classifier...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_val)
    dummy_metrics = evaluate(y_val, dummy_preds, CLASS_NAMES)

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_val)
    lr_metrics = evaluate(y_val, lr_preds, CLASS_NAMES)

    # 4. TRAIN XGBOOST
    logger.info("Training XGBoost Classifier...")
    # Convert string labels to integers for XGBoost (0, 1, 2)
    label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
    y_train_num = y_train.map(label_map)
    y_val_num = y_val.map(label_map)
    
    xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)
    xgb.fit(X_train, y_train_num)
    
    # Predict and map integers back to string labels
    xgb_preds_num = xgb.predict(X_val)
    inv_label_map = {i: name for name, i in label_map.items()}
    xgb_preds = [inv_label_map[p] for p in xgb_preds_num]
    xgb_metrics = evaluate(y_val, xgb_preds, CLASS_NAMES)

    # 5. PRINT COMPARISON
    print("\n--- MODEL COMPARISON (Macro-F1) ---")
    print(f"Dummy Classifier:    {dummy_metrics['macro_f1']:.4f}")
    print(f"Logistic Regression: {lr_metrics['macro_f1']:.4f}")
    print(f"XGBoost:             {xgb_metrics['macro_f1']:.4f}")

    # 6. SAVE ARTIFACTS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "triage_model.json"
    xgb.save_model(model_path)
    
    metadata = {
        "model_version": "0.1.0",
        "trained_on": data_source,
        "split_method": "time-based, incident-grouped",
        "timestamp": datetime.utcnow().isoformat(),
        "row_count": len(df),
        "model_type": "XGBClassifier",
        "macro_f1": float(xgb_metrics["macro_f1"]), 
        "feature_names": feature_cols,
        "class_names": CLASS_NAMES
    }
    
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved XGBoost model to {model_path} and metadata to metadata.json")

if __name__ == "__main__":
    main()