import polars as pl
from asoc_ml.config import ID_COL, ORG_COL, LABEL_COL, TIME_COL

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates raw evidence rows into a single row per incident.
    """
    if df.is_empty():
        return pl.DataFrame()

    # We group ONLY by IncidentId to ensure one row per incident
    aggregated = df.group_by(ID_COL).agg([
        pl.col(ORG_COL).first().alias("org_id"),
        
        # Take the first label found. If an incident has conflicting labels, 
        # taking the first is our temporary v0 policy.
        pl.col(LABEL_COL).first().alias("verdict"),
        
        # Extract the hour from the very first timestamp of the incident
        pl.col(TIME_COL).min().alias("incident_start_time"),
        pl.col(TIME_COL).min().dt.hour().alias("hour_of_day"),
        
        # Basic counts
        pl.len().alias("alert_count"),
        
        # Using real Kaggle columns: count distinct Mitre techniques and entities
        pl.col("MitreTechniques").drop_nulls().n_unique().alias("distinct_mitre_techniques"),
        pl.col("EntityType").drop_nulls().n_unique().alias("distinct_entity_types")
    ])
    
    return aggregated