"""Tests for incident splitting logic."""

from asoc_ml.data.load import make_synthetic_incidents
from asoc_ml.data.splits import make_splits


def test_no_incident_appears_in_more_than_one_split() -> None:
    df = make_synthetic_incidents(n=500, seed=7)
    train_df, val_df, test_df = make_splits(df, "Timestamp", "IncidentId")

    train_ids = train_df["IncidentId"].to_list()
    val_ids = val_df["IncidentId"].to_list()
    test_ids = test_df["IncidentId"].to_list()

    all_ids = train_ids + val_ids + test_ids
    assert len(all_ids) == len(set(all_ids))


def test_train_period_precedes_test_period() -> None:
    df = make_synthetic_incidents(n=500, seed=7)
    train_df, _, test_df = make_splits(df, "Timestamp", "IncidentId")

    assert train_df["Timestamp"].max() <= test_df["Timestamp"].min()
