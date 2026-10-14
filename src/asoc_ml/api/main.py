"""FastAPI service for the ASOC triage model."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from asoc_ml.api.schemas import (
    FeatureImpact,
    PredictRequest,
    PredictResponse,
    ProbabilityDistribution,
)

load_dotenv()

MOCK_MODE = os.getenv("MOCK", "false").lower() == "true"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/triage_model.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")

app = FastAPI(
    title="ASOC ML Triage API",
    version=MODEL_VERSION,
    description="Starter triage API for incident grading tasks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health probe for deployment and service checks."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Return version metadata for the serving model."""
    return {
        "model_version": MODEL_VERSION,
        "mock_mode": MOCK_MODE,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Return a triage prediction for a single incident."""
    if MOCK_MODE:
        base_probabilities = {
            "TruePositive": 0.62,
            "BenignPositive": 0.22,
            "FalsePositive": 0.16,
        }
        if request.alert_count >= 25:
            base_probabilities = {
                "TruePositive": 0.74,
                "BenignPositive": 0.15,
                "FalsePositive": 0.11,
            }
        verdict = max(base_probabilities, key=base_probabilities.get)
        confidence = base_probabilities[verdict]
        return PredictResponse(
            incident_id=request.incident_id,
            verdict=verdict,
            probabilities=ProbabilityDistribution(**base_probabilities),
            confidence=confidence,
            top_features=[
                FeatureImpact(name="alert_count", impact=0.32),
                FeatureImpact(name="n_devices", impact=0.18),
                FeatureImpact(name="distinct_mitre_techniques", impact=0.12),
            ],
            model_version=MODEL_VERSION,
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Train the model before serving with MOCK=false.",
        )

    from asoc_ml.models.predict import predict_one

    verdict, probabilities, confidence, top_features = predict_one(request.model_dump())
    return PredictResponse(
        incident_id=request.incident_id,
        verdict=verdict,
        probabilities=ProbabilityDistribution(**probabilities),
        confidence=confidence,
        top_features=[FeatureImpact(**feature) for feature in top_features],
        model_version=MODEL_VERSION,
    )
