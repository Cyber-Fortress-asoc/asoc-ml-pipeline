"""Time-aware split logic for incident-level training."""

from __future__ import annotations

import polars as pl


def make_splits(
    df: pl.DataFrame,
    time_col: str,
    group_col: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split incidents into train/validation/test by earliest timestamp.

    The method first computes the earliest timestamp for each incident, then orders
    incidents chronologically and slices the list into train/validation/test partitions.
    This prevents training rows from containing future information and avoids leakage
    across time windows.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")

    incident_times = (
        df.select([group_col, time_col])
        .unique()
        .group_by(group_col)
        .agg(pl.col(time_col).min().alias("_earliest_time"))
        .sort("_earliest_time")
    )

    ordered_ids = incident_times.get_column(group_col).to_list()
    total_incidents = len(ordered_ids)
    train_n = int(total_incidents * train_frac)
    val_n = int(total_incidents * val_frac)
    test_n = total_incidents - train_n - val_n

    if test_n <= 0:
        raise ValueError("Not enough incidents left for the test split.")

    train_ids = ordered_ids[:train_n]
    val_ids = ordered_ids[train_n : train_n + val_n]
    test_ids = ordered_ids[train_n + val_n :]

    train_df = df.filter(pl.col(group_col).is_in(train_ids))
    val_df = df.filter(pl.col(group_col).is_in(val_ids))
    test_df = df.filter(pl.col(group_col).is_in(test_ids))

    return train_df, val_df, test_df
