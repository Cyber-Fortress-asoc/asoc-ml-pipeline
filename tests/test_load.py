import polars as pl

from asoc_ml.data.load import load_data


def test_load_data_drops_null_labels_with_optional_column_selection(tmp_path) -> None:
    parquet_path = tmp_path / "incidents.parquet"
    pl.DataFrame(
        {
            "IncidentGrade": ["TruePositive", None, "FalsePositive"],
            "incident_id": [1, 2, 3],
        }
    ).write_parquet(parquet_path)

    loaded = load_data(parquet_path)
    selected = load_data(parquet_path, columns=["incident_id"])

    assert loaded["IncidentGrade"].to_list() == ["TruePositive", "FalsePositive"]
    assert selected["incident_id"].to_list() == [1, 3]
