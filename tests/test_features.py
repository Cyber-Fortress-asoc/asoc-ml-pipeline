import polars as pl

from asoc_ml.config import INCIDENT_START_TIME_COL, LABEL_COL
from asoc_ml.features.build import build_features


def test_build_features_preserves_incident_start_time() -> None:
    raw = pl.DataFrame(
        {
            "IncidentId": [1, 1, 2],
            "OrgId": [10, 10, 20],
            "IncidentGrade": ["TruePositive", "TruePositive", "FalsePositive"],
            "Timestamp": [
                "2024-06-10T13:30:56.000Z",
                "2024-06-10T14:30:56.000Z",
                "2024-06-11T19:47:36.000Z",
            ],
            "IpAddress": [100, 101, 200],
        }
    )

    features = build_features(raw)

    assert INCIDENT_START_TIME_COL in features.columns
    assert LABEL_COL in features.columns
    assert features.filter(pl.col("IncidentId") == 1)[
        INCIDENT_START_TIME_COL
    ].item().isoformat() == "2024-06-10T13:30:56+00:00"
