"""Pydantic models for the agent's state, plans, and outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: int
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class Plan(BaseModel):
    steps: list[PlanStep]
    reasoning: str = ""


class ToolCallResult(BaseModel):
    step_id: int
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: str = ""


class RetrievedChunkRef(BaseModel):
    chunk_id: str
    source: str = ""
    title: str = ""
    snippet: str = ""


class StrategicItem(BaseModel):
    title: str
    description: str
    impact: str = "Medium"
    confidence: float = 0.5
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    title: str
    rationale: str = ""
    priority: str = "Medium"
    expected_impact: str = ""
    risk_assessment: str = ""
    confidence: float = 0.5
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    supporting_evidence_chunk_ids: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    goal: str
    run_id: str = ""

    plan: Plan | None = None
    replan_count: int = 0

    next_step_index: int = 0
    tool_results: list[ToolCallResult] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkRef] = Field(default_factory=list)

    opportunities: list[StrategicItem] = Field(default_factory=list)
    risks: list[StrategicItem] = Field(default_factory=list)
    trends: list[StrategicItem] = Field(default_factory=list)

    analysis: AnalysisOutput | None = None

    recommendations: list[Recommendation] = Field(default_factory=list)
    briefing: str = ""

    validation: ValidationResult | None = None

    errors: list[str] = Field(default_factory=list)
