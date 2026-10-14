"""Pydantic schemas for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProbabilityDistribution(BaseModel):
    """Three-class probability distribution used by the triage model."""

    model_config = ConfigDict(extra="forbid")

    TruePositive: float = Field(ge=0.0, le=1.0)
    BenignPositive: float = Field(ge=0.0, le=1.0)
    FalsePositive: float = Field(ge=0.0, le=1.0)


class FeatureImpact(BaseModel):
    """Represents the impact of a single feature on the decision."""

    model_config = ConfigDict(extra="forbid")

    name: str
    impact: float = Field(ge=0.0)


class PredictRequest(BaseModel):
    """Request schema for incident triage predictions."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    org_id: int
    alert_count: int = Field(ge=0)
    distinct_mitre_techniques: int = Field(ge=0)
    distinct_entity_types: int = Field(ge=0)
    distinct_categories: int = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    n_devices: int = Field(ge=0)
    n_accounts: int = Field(ge=0)
    n_ips: int = Field(ge=0)


class PredictResponse(BaseModel):
    """Response schema returned after a prediction call."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    verdict: Literal["TruePositive", "BenignPositive", "FalsePositive"]
    probabilities: ProbabilityDistribution
    confidence: float = Field(ge=0.0, le=1.0)
    top_features: list[FeatureImpact]
    model_version: str
