"""Utilities for loading raw and synthetic incident data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from asoc_ml.config import ID_COL, LABEL_COL, ORG_COL, TIME_COL


def load_data(path: str | Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Load a Parquet file with Polars and return a collected DataFrame.

    Args:
        path: Location of the parquet file.
        columns: Optional subset of columns to read.

    Returns:
        A Polars DataFrame.
    """
    parquet_path = Path(path)
    lazy_frame = pl.scan_parquet(parquet_path)
    if columns is not None:
        lazy_frame = lazy_frame.select(columns)
    return lazy_frame.collect()


def make_synthetic_incidents(n: int = 2000, seed: int = 42) -> pl.DataFrame:
    """Create synthetic incident records for local development and tests.

    The DataFrame mirrors the key GUIDE-style fields so team members can prototype
    feature generation and split logic without a real dataset.
    """
    rng = np.random.default_rng(seed)
    categories = ["Phishing", "Malware", "Credential Abuse", "Suspicious Login", "Cloud"]
    techniques = [
        "T1059",
        "T1078",
        "T1110",
        "T1546",
        "T1566",
        "T1583",
        "T1602",
        "T1133",
    ]
    entity_types = ["User", "Device", "Endpoint", "MailBox", "CloudAccount", "Server"]
    labels = ["TruePositive", "BenignPositive", "FalsePositive"]

    incident_ids = [f"INC-{idx:06d}" for idx in range(n)]
    org_ids = rng.integers(1, 50, size=n).tolist()
    timestamps = [
        datetime(2023, 1, 1) + timedelta(days=int(value), hours=int((value % 1) * 24))
        for value in rng.uniform(0, 365, size=n)
    ]
    categories_list = [
        categories[int(value)] for value in rng.integers(0, len(categories), size=n)
    ]
    mitre_list = [techniques[int(value)] for value in rng.integers(0, len(techniques), size=n)]
    entity_list = [
        entity_types[int(value)] for value in rng.integers(0, len(entity_types), size=n)
    ]
    grades = [labels[int(value)] for value in rng.integers(0, len(labels), size=n)]

    return pl.DataFrame(
        {
            ID_COL: incident_ids,
            ORG_COL: org_ids,
            TIME_COL: timestamps,
            "Category": categories_list,
            "MitreTechniques": mitre_list,
            "EntityType": entity_list,
            LABEL_COL: grades,
            "n_devices": rng.integers(1, 25, size=n).tolist(),
            "n_accounts": rng.integers(1, 18, size=n).tolist(),
            "n_ips": rng.integers(1, 12, size=n).tolist(),
        }
    )
