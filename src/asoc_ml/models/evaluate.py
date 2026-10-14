"""Evaluation helpers for model comparison."""

from __future__ import annotations

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate(y_true, y_pred, class_names):
    """Compute confusion matrix, per-class metrics, and macro-F1.

    Accuracy is a useful sanity check, but it is not the headline metric for this
    incident triage task because class imbalance can mask poor minority-class
    performance.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        average=None,
        zero_division=0,
    )

    per_class = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for name, p, r, f in zip(class_names, precision, recall, f1, strict=False)
    }
    macro_f1 = float(
        sum(f1_values["f1"] for f1_values in per_class.values()) / len(class_names)
    )

    results = {
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_f1": macro_f1,
    }

    print("Confusion matrix:")
    print(cm)
    print("Per-class metrics:")
    for name in class_names:
        metrics = per_class[name]
        print(
            f"  {name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )
    print(f"Macro-F1: {macro_f1:.4f}")

    return results
