# Prediction API Contract

This service predicts the likely incident grade for a new security alert bundle. The model is intentionally simple at v0 and is designed for integration with the backend, agent, and frontend teams while the EDA is still in progress.

## Endpoint

POST /predict

## Request body

The request body includes a minimal set of incident-level fields that are expected to be available at inference time. These are placeholder v0 fields and will change once the EDA is complete.

Example request:

```json
{
  "incident_id": "INC-2024-00042",
  "org_id": 17,
  "alert_count": 18,
  "distinct_mitre_techniques": 4,
  "distinct_entity_types": 3,
  "distinct_categories": 2,
  "hour_of_day": 22,
  "n_devices": 17,
  "n_accounts": 8,
  "n_ips": 6
}
```

### Field definitions

- incident_id: unique incident identifier as a string
- org_id: integer organization identifier
- alert_count: number of alerts associated with the investigation
- distinct_mitre_techniques: count of distinct MITRE ATT&CK techniques observed
- distinct_entity_types: count of distinct entity types involved
- distinct_categories: number of different categories in the alert bundle
- hour_of_day: hour of first alert or incident creation time, between 0 and 23
- n_devices: count of devices associated with the incident
- n_accounts: count of accounts associated with the incident
- n_ips: count of IP addresses involved

## Response body

Example response:

```json
{
  "incident_id": "INC-2024-00042",
  "verdict": "TruePositive",
  "probabilities": {
    "TruePositive": 0.62,
    "BenignPositive": 0.21,
    "FalsePositive": 0.17
  },
  "confidence": 0.62,
  "top_features": [
    {"name": "alert_count", "impact": 0.31},
    {"name": "n_devices", "impact": 0.18},
    {"name": "n_accounts", "impact": 0.12}
  ],
  "model_version": "0.1.0"
}
```

### Response field definitions

- incident_id: same identifier passed in the request
- verdict: predicted incident grade; one of `TruePositive`, `BenignPositive`, or `FalsePositive`
- probabilities: class-wise probability distribution for all three classes
- confidence: highest predicted probability for the chosen verdict
- top_features: ordered list of feature names and their estimated impact on the prediction
- model_version: model version string from the model metadata

## Notes

- The current v0 contract is intentionally minimal and will evolve as EDA is completed.
- The feature list is not final and should be validated against the real GUIDE data before production use.

## Open questions

- Should the platform predict at the alert-level or the incident-level?
- Should `BenignPositive` and `FalsePositive` be collapsed into one class, or kept distinct for triage?
- Should the feature set include behavior-derived features measured only after the incident is already known?
