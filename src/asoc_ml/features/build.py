import polars as pl

from asoc_ml.config import (
    ID_COL,
    INCIDENT_START_TIME_COL,
    LABEL_COL,
    ORG_COL,
    TIME_COL,
)


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates raw evidence rows into a single row per incident.
    """
    if df.is_empty():
        return pl.DataFrame()

    df = df.with_columns(
        pl.col(TIME_COL).str.to_datetime(format="%+", strict=False).alias("parsed_time")
    )
    # We group ONLY by IncidentId to ensure one row per incident
    aggregated = df.group_by(ID_COL).agg([
        pl.col(ORG_COL).first().alias("org_id"),
        
        # Take the first label found. If an incident has conflicting labels, 
        # taking the first is our temporary v0 policy.
        pl.col(LABEL_COL).first().alias(LABEL_COL),
        
        # --- TIME FEATURES ---
        pl.col("parsed_time").min().alias(INCIDENT_START_TIME_COL),
        pl.col("parsed_time").min().dt.hour().alias("hour_of_day"),
        # Duration in seconds between the first and last evidence row
        (pl.col("parsed_time").max() - pl.col("parsed_time").min())
        .dt.total_seconds()
        .alias("incident_duration_sec"),
        # ---------------------

        # --- NETWORK FEATURES ---
        # Drop nulls before counting distinct IP addresses
        pl.col("IpAddress").drop_nulls().n_unique().alias("distinct_ips"),
        # ------------------------

        # Basic counts
        pl.len().alias("alert_count"),

        # --------------------------- TODO: STUDENT 1 (UPDATE) ---------------------------
        # Using real Kaggle columns: count distinct Mitre techniques and entities
        # pl.col("MitreTechniques").drop_nulls().n_unique().alias("distinct_mitre_techniques"),
        # pl.col("EntityType").drop_nulls().n_unique().alias("distinct_entity_types")
    ])
    
    return aggregated