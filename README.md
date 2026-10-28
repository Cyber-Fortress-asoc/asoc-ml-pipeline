# ASOC ML Pipeline

This repository contains the machine-learning layer for an Autonomous AI SOC Analyst / XDR triage platform. It trains a classifier on Microsoft GUIDE-style incident records to predict the incident verdict (`TruePositive`, `BenignPositive`, `FalsePositive`) and exposes the model through a FastAPI service for the backend, agent, and frontend teams.

## Project goals

- Keep the pipeline beginner-friendly and reproducible
- Prevent data leakage by using time-aware incident-level splitting
- Separate exploration from reusable production code (`notebooks/` for experiments, `src/` for buildable code)
- Keep API contracts explicit in `contract/`

## Phase 1: ML Triage & Detection Layer

This repository handles the data ingestion, feature engineering, and incident triage for the ASOC platform. It consumes raw security evidence rows (from Microsoft GUIDE), aggregates them into discrete incidents, and uses machine learning to predict the incident's legitimacy.

### System Design & Implementation
* **Data Engine (Polars):** We use Polars for lazy-loading and out-of-core memory execution to process multi-gigabyte Parquet files without exhausting system RAM.
* **Incident Aggregation:** Raw telemetry rows are grouped by `IncidentId`. Time-series features (like `hour_of_day` and `incident_duration`) are calculated from strictly parsed ISO-8601 timestamps to handle Microsoft's UTC format.
* **Leakage-Proof Validation:** The pipeline utilizes a strict temporal, incident-grouped split. Incidents are sorted chronologically by their earliest timestamp. Train, validation, and test splits never straddle the same incident, preventing temporal data leakage.
* **Modeling (XGBoost):** Due to the highly imbalanced and non-linear nature of security logs, we utilize XGBoost with sample weights. It outperforms baseline Logistic Regression by over 2x (Macro-F1) on base features.

## Repository layout

```text
asoc-ml-pipeline/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── .github/
├── contract/
├── src/
│   └── asoc_ml/
├── tests/
├── notebooks/
├── docs/
├── data/
├── models/
└── .gitignore
```

## Quickstart

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or .venv\Scripts\activate  # Windows PowerShell
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

3. Run tests:

   ```bash
   make test
   ```

4. Train a baseline model:

   ```bash
   make train
   ```

5. Start the API locally with a mock response mode:

   ```bash
   make serve
   ```

   Then open http://localhost:8000/docs

## Notebook workflow

- Notebook work is for exploration and prototyping only.
- One notebook = one owner.
- Use Jupytext paired notebooks in `notebooks/`.
- If code becomes reusable, move it into `src/asoc_ml/`.
- Pull requests must be reviewed by the other student.

## Contract

The API contract is defined in [contract/contract.md](contract/contract.md) and [contract/schema.json](contract/schema.json).

## Notes

- Do not commit raw datasets, secrets, or notebook outputs.
- Prefer time-based incident grouping when creating splits to avoid leakage.
- Keep feature engineering explicit and documented.
