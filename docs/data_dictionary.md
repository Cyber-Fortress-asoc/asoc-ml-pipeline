# Data dictionary

| Column | Type | Meaning | Missing % | Leakage risk? |
| --- | --- | --- | --- | --- |
| IncidentId | string | Unique identifier for an incident | 0% | Low |
| Timestamp | datetime | Event time for the incident or alert | 0% | Medium |
| OrgId | integer | Organization identifier | 0% | Low |
| IncidentGrade | string | Target label: TruePositive / BenignPositive / FalsePositive | 0% | Low |
| Category | string | Category extracted from the alert | TBD | Medium |
| MitreTechniques | string | Associated ATT&CK technique identifier | TBD | Medium |
| EntityType | string | Entity type involved in the alert | TBD | Medium |
| ActionGrouped | string | Potentially post-incident derived field | TBD | High |
| ActionGranular | string | Potentially post-incident derived field | TBD | High |

Update this table after the EDA is complete.
