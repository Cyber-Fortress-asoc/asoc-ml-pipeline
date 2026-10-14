"""Feature-building utilities for the ASOC incident triage dataset.

The current v0 feature set is intentionally small and transparent. Student A should
extend this module with richer temporal and behavioral features after EDA is complete.
"""

from __future__ import annotations

import polars as pl

from asoc_ml.config import ID_COL, LABEL_COL, ORG_COL, TIME_COL


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw alert rows to a single row per incident.

    v0 feature columns:
    - incident_id: unique incident identifier
    - org_id: organization identifier
    - alert_count: total number of alerts associated with the incident
    - distinct_mitre_techniques: number of unique MITRE ATT&CK techniques reported
    - distinct_entity_types: number of unique entity types involved
    - distinct_categories: number of distinct alert categories
    - hour_of_day: approximate hour of first observed activity
    - n_devices: number of devices recorded for the incident
    - n_accounts: number of accounts involved
    - n_ips: number of IP addresses involved
    - IncidentGrade: target label

    TODO: Student A should add better temporal features, entity-risk features,
    alert severity summaries, and anomaly counts once the EDA is finished.
    """
    if df.is_empty():
        return pl.DataFrame(
            schema={
                ID_COL: pl.String,
                ORG_COL: pl.Int64,
                TIME_COL: pl.Datetime,
                "alert_count": pl.Int64,
                "distinct_mitre_techniques": pl.Int64,
                "distinct_entity_types": pl.Int64,
                "distinct_categories": pl.Int64,
                "hour_of_day": pl.Int64,
                "n_devices": pl.Int64,
                "n_accounts": pl.Int64,
                "n_ips": pl.Int64,
                LABEL_COL: pl.String,
            }
        )

    grouped = (
        df.with_columns(
            pl.col(TIME_COL).cast(pl.Datetime).alias(TIME_COL),
            pl.col(TIME_COL).dt.hour().cast(pl.Int64).alias("hour_of_day"),
            pl.col("MitreTechniques").fill_null("Unknown").cast(pl.String),
            pl.col("EntityType").fill_null("Unknown").cast(pl.String),
            pl.col("Category").fill_null("Unknown").cast(pl.String),
        )
        .group_by([ID_COL, ORG_COL, LABEL_COL])
        .agg(
            pl.len().alias("alert_count"),
            pl.col(TIME_COL).min().alias(TIME_COL),
            pl.col("MitreTechniques").n_unique().alias("distinct_mitre_techniques"),
            pl.col("EntityType").n_unique().alias("distinct_entity_types"),
            pl.col("Category").n_unique().alias("distinct_categories"),
            pl.col("hour_of_day").mean().round(0).cast(pl.Int64).alias("hour_of_day"),
            pl.col("n_devices").sum().alias("n_devices"),
            pl.col("n_accounts").sum().alias("n_accounts"),
            pl.col("n_ips").sum().alias("n_ips"),
        )
        .sort(ID_COL)
    )

    return grouped.rename({ID_COL: "incident_id", ORG_COL: "org_id"})
