# Model card

## Intended use

This model is intended as a starter incident-grade classifier for a SOC triage platform. It is designed to support early experimentation and integration work while the project team completes EDA and feature validation.

## Data

- Source: Microsoft GUIDE-style security incident dataset
- Label: `IncidentGrade`
- Current v0 feature set: organization id, alert count, MITRE technique count, entity count, category count, hour-of-day, device count, account count, IP count
- Dataset handling: no raw data is committed to this repository

## Split method

The split is time-based and incident-grouped to reduce leakage from future-time events. Each incident is assigned to only one split according to its earliest observed time.

## Metrics

- Primary metric: macro-F1
- Supporting metrics: per-class precision, recall, and F1, plus the confusion matrix
- Reporting should include minority-class performance because the dataset is likely imbalanced

## Limitations

- This is a starter model with a small v0 feature set
- The feature engineering is intentionally simple and may not capture all alert context
- The dataset schema may change after EDA completes
- The model should not be treated as production-grade until the raw data contract and leakage audit are complete

## Version history

| Version | Summary | Date |
| --- | --- | --- |
| 0.1.0 | Initial baseline with synthetic fallback and basic API contract | 2026-10-14 |
