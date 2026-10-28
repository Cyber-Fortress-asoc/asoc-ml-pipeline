import polars as pl
from asoc_ml.data.splits import make_splits


def _make_toy_incidents() -> pl.DataFrame:
    """
    Creates a tiny fake dataset with 6 incidents (IDs 1 through 6).
    Notice that Incident 1 has 3 rows, Incident 2 has 2 rows, and Incident 4 has 2 rows.
    Notice also that Incident 1 starts at Timestamp=1, but has a late row at Timestamp=9!
    """
    return pl.DataFrame({
        "IncidentId": [1, 1, 1,  2, 2,  3,  4, 4,  5,  6],
        "Timestamp":  [1, 5, 9,  2, 3,  4,  6, 7,  8, 10],
    })


def test_incidents_never_straddle_splits():
    """
    Rule A Check: Verify that all rows belonging to a single IncidentId
    end up in the exact same split (Train, Val, or Test).
    """
    df = _make_toy_incidents()
    train_df, val_df, test_df = make_splits(df, time_col="Timestamp", group_col="IncidentId")

    # Get the unique set of IncidentIds in each split
    train_ids = set(train_df["IncidentId"])
    val_ids = set(val_df["IncidentId"])
    test_ids = set(test_df["IncidentId"])

    # .isdisjoint() checks that two sets share ZERO common elements
    assert train_ids.isdisjoint(val_ids), "Bug: An incident appeared in both Train and Val!"
    assert train_ids.isdisjoint(test_ids), "Bug: An incident appeared in both Train and Test!"
    assert val_ids.isdisjoint(test_ids), "Bug: An incident appeared in both Val and Test!"


def test_no_rows_lost():
    """
    Sanity Check: Verify that splitting the table doesn't accidentally delete any rows.
    """
    df = _make_toy_incidents()
    train_df, val_df, test_df = make_splits(df, time_col="Timestamp", group_col="IncidentId")

    assert train_df.height + val_df.height + test_df.height == df.height


def test_incident_start_times_are_chronological():
    """
    Rule B Check: Verify that every incident in Train started BEFORE (or at the same time as)
    every incident in Test.
    """
    df = _make_toy_incidents()
    train_df, _, test_df = make_splits(df, time_col="Timestamp", group_col="IncidentId")

    # Find the earliest (min) Timestamp for each incident in Train and Test
    train_start_times = train_df.group_by("IncidentId").agg(pl.col("Timestamp").min())["Timestamp"]
    test_start_times = test_df.group_by("IncidentId").agg(pl.col("Timestamp").min())["Timestamp"]

    # The latest start time in Train must be <= the earliest start time in Test
    assert train_start_times.max() <= test_start_times.min()