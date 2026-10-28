"""Configuration constants for the ASOC ML pipeline.

The real GUIDE dataset naming conventions should be confirmed in EDA before the
model is promoted beyond the v0 starter setup.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ID_COL = "IncidentId"
TIME_COL = "Timestamp"
INCIDENT_START_TIME_COL = "incident_start_time"
ORG_COL = "OrgId"
LABEL_COL = "IncidentGrade"

CLASS_NAMES = ["TruePositive", "BenignPositive", "FalsePositive"]
LEAKAGE_SUSPECTS = ["ActionGrouped", "ActionGranular"]
# Unverified: confirm in EDA whether these exist at alert time; if not, exclude them.
# Confirm all column names against the real files before moving beyond the starter pipeline.
