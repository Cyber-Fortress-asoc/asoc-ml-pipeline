"""Example tests for model evaluation metrics."""

from asoc_ml.models.evaluate import evaluate


def test_macro_f1_matches_expected_value() -> None:
    y_true = ["A", "A", "A", "B", "B", "B"]
    y_pred = ["A", "A", "B", "B", "B", "A"]

    result = evaluate(y_true, y_pred, ["A", "B"])

    assert abs(result["macro_f1"] - (2 / 3)) < 1e-9
