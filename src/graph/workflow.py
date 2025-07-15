"""
LangGraph StateGraph implementing the prior authorization review workflow.

The graph models the clinical PA review process as a directed graph with
conditional edges. Each node corresponds to a discrete review step and
updates the shared AgentState.

Key design decisions:
- Eligibility is a hard gate — if the member isn't eligible, we short-circuit
  to denial without wasting tool calls on formulary/clinical checks.
- Formulary check happens before clinical criteria because it's cheaper and
  faster; no point running RAG if the drug isn't even on formulary (unless
  we're checking exceptions).
- Drug interactions run last because they need the full medication context
  and don't affect the formulary/clinical determination.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.state import (
    AgentState,
    ClinicalGuidelineResult,
    CostEstimate,
    EligibilityResult,
    FormularyResult,
    InteractionResult,
    InteractionSeverity,
    PADecision,
    PADecisionType,
    ReviewStep,
)
from src.prompts.decision_criteria import evaluate_decision_rules
from src.tools.clinical_guidelines import clinical_guidelines
from src.tools.cost_estimator import cost_estimator
from src.tools.drug_interaction import drug_interaction
from src.tools.eligibility_check import eligibility_check
from src.tools.formulary_lookup import formulary_lookup
from src.tools.icd10_validator import icd10_validator

logger = logging.getLogger(__name__)


def _parse_json(raw: str) -> dict[str, Any]:
    """Safely parse JSON from tool output."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"error": f"Failed to parse tool output: {raw[:200]}"}


# ── Node functions ──────────────────────────────────────────────────────────


def check_eligibility(state: dict) -> dict:
    """Verify member plan eligibility and pharmacy benefit status."""
    agent_state = AgentState(**state)
    member_id = agent_state.request.member_id

    raw = eligibility_check.invoke(member_id)
    parsed = _parse_json(raw)
    result = EligibilityResult(**parsed)

    step = ReviewStep(
        step_name="check_eligibility",
        tool_called="eligibility_check",
        tool_input={"member_id": member_id},
        tool_output=parsed,
        reasoning=(
            f"Member {member_id}: plan_active={result.plan_active}, "
            f"pharmacy_benefit={result.pharmacy_benefit_active}"
        ),
    )

    return {
        **state,
        "eligibility": result.model_dump(),
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["check_eligibility"],
    }


def lookup_formulary(state: dict) -> dict:
    """Check drug formulary coverage, tier, and step therapy requirements."""
    agent_state = AgentState(**state)
    drug = agent_state.request.drug_name

    raw = formulary_lookup.invoke(drug)
    parsed = _parse_json(raw)
    result = FormularyResult(**parsed)

    step = ReviewStep(
        step_name="lookup_formulary",
        tool_called="formulary_lookup",
        tool_input={"drug_name": drug},
        tool_output=parsed,
        reasoning=(
            f"{drug}: tier={result.tier} ({result.tier_label}), "
            f"covered={result.covered}, pa_required={result.pa_required}, "
            f"step_therapy={result.step_therapy_drugs}"
        ),
    )

    return {
        **state,
        "formulary": result.model_dump(),
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["lookup_formulary"],
    }


def check_clinical_criteria(state: dict) -> dict:
    """Search clinical guidelines via RAG to determine if criteria are met."""
    agent_state = AgentState(**state)
    drug = agent_state.request.drug_name
    dx_codes = ",".join(agent_state.request.diagnosis_codes)

    raw = clinical_guidelines.invoke({"drug": drug, "diagnosis": dx_codes})
    parsed = _parse_json(raw)
    result = ClinicalGuidelineResult(**parsed)

    step = ReviewStep(
        step_name="check_clinical_criteria",
        tool_called="clinical_guidelines",
        tool_input={"drug": drug, "diagnosis": dx_codes},
        tool_output=parsed,
        reasoning=(
            f"Guideline: {result.guideline_source}, "
            f"criteria_met={result.criteria_met}, "
            f"relevance={result.relevance_score}"
        ),
    )

    return {
        **state,
        "clinical_criteria": result.model_dump(),
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["check_clinical_criteria"],
    }


def validate_icd10_codes(state: dict) -> dict:
    """Validate all submitted ICD-10 codes."""
    agent_state = AgentState(**state)
    codes = ",".join(agent_state.request.diagnosis_codes)

    raw = icd10_validator.invoke(codes)
    parsed = _parse_json(raw)

    all_valid = parsed.get("all_valid", False)
    code_details = parsed.get("codes", [])

    step = ReviewStep(
        step_name="validate_icd10_codes",
        tool_called="icd10_validator",
        tool_input={"codes": codes},
        tool_output=parsed,
        reasoning=f"ICD-10 validation: all_valid={all_valid}",
    )

    return {
        **state,
        "icd10_valid": all_valid,
        "icd10_details": code_details,
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["validate_icd10_codes"],
    }


def check_interactions(state: dict) -> dict:
    """Check drug-drug interactions between the new drug and current medications."""
    agent_state = AgentState(**state)
    drug = agent_state.request.drug_name
    current_meds = ",".join(agent_state.request.current_medications)

    if not current_meds:
        result = InteractionResult()
        step = ReviewStep(
            step_name="check_interactions",
            tool_called="drug_interaction",
            tool_input={"drug": drug, "current_medications": ""},
            tool_output={"interactions": [], "max_severity": "none"},
            reasoning="No current medications listed — no interactions to check.",
        )
    else:
        raw = drug_interaction.invoke({"drug": drug, "current_medications": current_meds})
        parsed = _parse_json(raw)
        result = InteractionResult(**parsed)
        step = ReviewStep(
            step_name="check_interactions",
            tool_called="drug_interaction",
            tool_input={"drug": drug, "current_medications": current_meds},
            tool_output=parsed,
            reasoning=(
                f"Max severity: {result.max_severity.value}, "
                f"{len(result.interactions)} interaction(s) checked"
            ),
        )

    return {
        **state,
        "interactions": result.model_dump(),
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["check_interactions"],
    }


def estimate_cost(state: dict) -> dict:
    """Estimate drug cost for the plan and member."""
    agent_state = AgentState(**state)
    drug = agent_state.request.drug_name
    qty = agent_state.request.quantity
    days = agent_state.request.days_supply

    raw = cost_estimator.invoke({"drug": drug, "quantity": qty, "days_supply": days})
    parsed = _parse_json(raw)
    result = CostEstimate(**parsed)

    step = ReviewStep(
        step_name="estimate_cost",
        tool_called="cost_estimator",
        tool_input={"drug": drug, "quantity": qty, "days_supply": days},
        tool_output=parsed,
        reasoning=(
            f"Plan cost: ${result.plan_cost_30day}/30 days, "
            f"member copay: ${result.member_copay}, "
            f"alternatives: {len(result.cheaper_alternatives)}"
        ),
    )

    return {
        **state,
        "cost_estimate": result.model_dump(),
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["estimate_cost"],
    }


def make_decision(state: dict) -> dict:
    """Apply decision rules to accumulated evidence and produce a PA decision."""
    agent_state = AgentState(**state)
    decision = evaluate_decision_rules(agent_state)

    step = ReviewStep(
        step_name="make_decision",
        tool_called="decision_criteria",
        tool_input={},
        tool_output=decision.model_dump(),
        reasoning=decision.rationale,
    )

    return {
        **state,
        "decision": decision.model_dump(),
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["make_decision"],
    }


def generate_summary(state: dict) -> dict:
    """Generate a human-readable summary of the PA review."""
    agent_state = AgentState(**state)
    decision = agent_state.decision

    if not decision:
        summary = "ERROR: No decision was made. Review incomplete."
    else:
        drug = agent_state.request.drug_name
        member = agent_state.request.member_id
        dx = ", ".join(agent_state.request.diagnosis_codes)

        lines = [
            f"═══ PRIOR AUTHORIZATION DECISION ═══",
            f"",
            f"Member: {member}",
            f"Drug: {drug}",
            f"Diagnosis: {dx}",
            f"",
            f"Decision: {decision.decision.upper()}",
            f"Confidence: {decision.confidence:.0%}",
            f"",
            f"Rationale:",
            f"  {decision.rationale}",
            f"",
            f"Evidence:",
        ]
        for ev in decision.cited_evidence:
            lines.append(f"  • {ev}")
        lines.extend([
            f"",
            f"Recommended Action:",
            f"  {decision.recommended_action}",
            f"",
            f"Human Review Required: {'Yes' if decision.requires_human_review else 'No'}",
            f"",
            f"─── Review Steps ({len(agent_state.review_steps)}) ───",
        ])
        for i, step_data in enumerate(agent_state.review_steps, 1):
            step = step_data if isinstance(step_data, ReviewStep) else ReviewStep(**step_data)
            lines.append(f"  {i}. [{step.step_name}] {step.reasoning}")

        summary = "\n".join(lines)

    step = ReviewStep(
        step_name="generate_summary",
        tool_called="",
        tool_input={},
        tool_output={"summary_length": len(summary)},
        reasoning="Generated final review summary.",
    )

    return {
        **state,
        "summary": summary,
        "review_steps": [s if isinstance(s, dict) else s.model_dump() for s in agent_state.review_steps] + [step.model_dump()],
        "completed_nodes": agent_state.completed_nodes + ["generate_summary"],
    }


# ── Conditional edge functions ──────────────────────────────────────────────


def route_after_eligibility(state: dict) -> str:
    """Route based on eligibility result — deny immediately if not eligible."""
    elig = state.get("eligibility", {})
    if not elig.get("plan_active", False) or not elig.get("pharmacy_benefit_active", False):
        return "make_decision"
    return "validate_icd10_codes"


def route_after_formulary(state: dict) -> str:
    """Route based on formulary result — skip clinical check if drug isn't covered."""
    form = state.get("formulary", {})
    if not form.get("covered", False):
        return "make_decision"
    return "check_clinical_criteria"


# ── Graph construction ──────────────────────────────────────────────────────


def build_pa_review_graph() -> StateGraph:
    """Construct and compile the prior authorization review workflow graph."""

    workflow = StateGraph(dict)

    workflow.add_node("check_eligibility", check_eligibility)
    workflow.add_node("validate_icd10_codes", validate_icd10_codes)
    workflow.add_node("lookup_formulary", lookup_formulary)
    workflow.add_node("check_clinical_criteria", check_clinical_criteria)
    workflow.add_node("check_interactions", check_interactions)
    workflow.add_node("estimate_cost", estimate_cost)
    workflow.add_node("make_decision", make_decision)
    workflow.add_node("generate_summary", generate_summary)

    workflow.set_entry_point("check_eligibility")

    workflow.add_conditional_edges(
        "check_eligibility",
        route_after_eligibility,
        {
            "make_decision": "make_decision",
            "validate_icd10_codes": "validate_icd10_codes",
        },
    )

    workflow.add_edge("validate_icd10_codes", "lookup_formulary")

    workflow.add_conditional_edges(
        "lookup_formulary",
        route_after_formulary,
        {
            "make_decision": "make_decision",
            "check_clinical_criteria": "check_clinical_criteria",
        },
    )

    workflow.add_edge("check_clinical_criteria", "check_interactions")
    workflow.add_edge("check_interactions", "estimate_cost")
    workflow.add_edge("estimate_cost", "make_decision")
    workflow.add_edge("make_decision", "generate_summary")
    workflow.add_edge("generate_summary", END)

    return workflow


def compile_graph():
    """Build and compile the graph, returning a runnable."""
    graph = build_pa_review_graph()
    return graph.compile()


pa_review_chain = compile_graph()
