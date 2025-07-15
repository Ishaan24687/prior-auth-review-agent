"""
Pydantic models for the prior authorization review graph state.

These models enforce the data contracts between workflow nodes. Every node
reads from and writes to a shared AgentState instance, so strict typing
here catches integration bugs early.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PADecisionType(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    PEND = "pend"


class UrgencyLevel(str, Enum):
    STANDARD = "standard"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class InteractionSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class PARequest(BaseModel):
    """Incoming prior authorization request — mirrors the fields a plan would
    receive from the provider's ePA submission (NCPDP SCRIPT standard)."""

    member_id: str = Field(..., description="Plan member identifier")
    drug_name: str = Field(..., description="Prescribed drug (brand or generic)")
    diagnosis_codes: list[str] = Field(
        default_factory=list,
        description="ICD-10-CM codes supporting medical necessity",
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="Member's active medication list",
    )
    provider_npi: str = Field(default="", description="Prescribing provider NPI")
    quantity: int = Field(default=30, description="Quantity requested")
    days_supply: int = Field(default=30, description="Days supply requested")
    urgency: UrgencyLevel = Field(default=UrgencyLevel.STANDARD)
    # TODO: add prior_treatments field for step therapy documentation


class EligibilityResult(BaseModel):
    plan_active: bool = False
    plan_type: str = ""
    effective_date: str = ""
    termination_date: str = ""
    pharmacy_benefit_active: bool = False
    member_name: str = ""
    error: str = ""


class FormularyResult(BaseModel):
    drug_name: str = ""
    generic_name: str = ""
    tier: int = 0
    tier_label: str = ""
    covered: bool = False
    pa_required: bool = False
    step_therapy_required: bool = False
    step_therapy_drugs: list[str] = Field(default_factory=list)
    quantity_limit: str = ""
    ndc_code: str = ""
    error: str = ""


class ClinicalGuidelineResult(BaseModel):
    guideline_text: str = ""
    criteria_met: bool = False
    guideline_source: str = ""
    relevance_score: float = 0.0
    error: str = ""


class DrugInteraction(BaseModel):
    drug_pair: str = ""
    severity: InteractionSeverity = InteractionSeverity.NONE
    description: str = ""


class InteractionResult(BaseModel):
    interactions: list[DrugInteraction] = Field(default_factory=list)
    max_severity: InteractionSeverity = InteractionSeverity.NONE
    error: str = ""


class CostEstimate(BaseModel):
    drug_name: str = ""
    plan_cost_30day: float = 0.0
    member_copay: float = 0.0
    member_coinsurance_pct: float = 0.0
    total_annual_cost: float = 0.0
    cheaper_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class ReviewStep(BaseModel):
    """Single step in the agent's review trace — stored so we can show the
    full reasoning chain in the Gradio UI and evaluation harness."""

    step_name: str
    tool_called: str = ""
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PADecision(BaseModel):
    decision: PADecisionType = PADecisionType.PEND
    confidence: float = 0.0
    rationale: str = ""
    cited_evidence: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    requires_human_review: bool = True


class AgentState(BaseModel):
    """Full state object that flows through the LangGraph workflow.
    Each node reads what it needs and appends its results."""

    request: PARequest
    eligibility: Optional[EligibilityResult] = None
    formulary: Optional[FormularyResult] = None
    clinical_criteria: Optional[ClinicalGuidelineResult] = None
    interactions: Optional[InteractionResult] = None
    cost_estimate: Optional[CostEstimate] = None
    icd10_valid: bool = True
    icd10_details: list[dict[str, Any]] = Field(default_factory=list)
    review_steps: list[ReviewStep] = Field(default_factory=list)
    decision: Optional[PADecision] = None
    summary: str = ""
    error: Optional[str] = None
    # track which nodes have executed for conditional routing
    completed_nodes: list[str] = Field(default_factory=list)
