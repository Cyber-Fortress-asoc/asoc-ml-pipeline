"""Integration tests for the FastAPI layer."""

import os

from fastapi.testclient import TestClient

os.environ["MOCK"] = "true"

from asoc_ml.api.main import app


def test_health_and_mock_prediction() -> None:
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    payload = {
        "incident_id": "INC-00001",
        "org_id": 7,
        "alert_count": 12,
        "distinct_mitre_techniques": 2,
        "distinct_entity_types": 3,
        "distinct_categories": 2,
        "hour_of_day": 18,
        "n_devices": 7,
        "n_accounts": 4,
        "n_ips": 4,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["incident_id"] == payload["incident_id"]
    assert result["verdict"] in ["TruePositive", "BenignPositive", "FalsePositive"]

    invalid_payload = dict(payload)
    invalid_payload.pop("org_id")
    invalid_response = client.post("/predict", json=invalid_payload)
    assert invalid_response.status_code == 422
