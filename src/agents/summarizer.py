"""
PA decision summarizer.

Takes the full review trace (tool calls, observations, decision) and
generates a human-readable summary suitable for:
  1. Provider notification letters
  2. Member explanation of benefits
  3. Internal audit trail
  4. Compliance documentation

Can operate in two modes:
  - LLM mode: uses GPT-4 to generate a natural language summary
  - Template mode: uses a deterministic template (no LLM calls, cheaper)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from src.graph.state import AgentState, PADecisionType, ReviewStep

logger = logging.getLogger(__name__)


def generate_template_summary(state: AgentState) -> str:
    """Generate a deterministic template-based summary without LLM calls.
    This is the default mode — fast, cheap, and consistent."""

    decision = state.decision
    if not decision:
        return "Review incomplete — no decision was reached."

    request = state.request
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    decision_label = {
        PADecisionType.APPROVE: "APPROVED",
        PADecisionType.DENY: "DENIED",
        PADecisionType.PEND: "PENDED FOR HUMAN REVIEW",
    }[decision.decision]

    sections = []

    sections.append(
        f"PRIOR AUTHORIZATION DETERMINATION\n"
        f"{'=' * 40}\n"
        f"Date: {now}\n"
        f"Reference: PA-{request.member_id}-{request.drug_name.upper()[:6]}\n"
    )

    sections.append(
        f"REQUEST DETAILS\n"
        f"  Member ID: {request.member_id}\n"
        f"  Drug: {request.drug_name}\n"
        f"  Diagnosis: {', '.join(request.diagnosis_codes)}\n"
        f"  Current Medications: {', '.join(request.current_medications) or 'None listed'}\n"
        f"  Provider NPI: {request.provider_npi or 'Not provided'}\n"
        f"  Quantity: {request.quantity} units / {request.days_supply} days\n"
        f"  Urgency: {request.urgency.value}\n"
    )

    sections.append(f"DETERMINATION: {decision_label}\n")

    # Eligibility section
    if state.eligibility:
        elig = state.eligibility
        elig_status = "Active" if elig.plan_active else "INACTIVE"
        rx_status = "Active" if elig.pharmacy_benefit_active else "INACTIVE"
        sections.append(
            f"ELIGIBILITY\n"
            f"  Plan Status: {elig_status} ({elig.plan_type})\n"
            f"  Effective: {elig.effective_date} — {elig.termination_date}\n"
            f"  Pharmacy Benefit: {rx_status}\n"
        )

    # Formulary section
    if state.formulary:
        form = state.formulary
        sections.append(
            f"FORMULARY STATUS\n"
            f"  Coverage: {'Covered' if form.covered else 'NOT COVERED'}\n"
            f"  Tier: {form.tier} ({form.tier_label})\n"
            f"  PA Required: {'Yes' if form.pa_required else 'No'}\n"
            f"  Step Therapy: {'Required — ' + ', '.join(form.step_therapy_drugs) if form.step_therapy_required else 'Not required'}\n"
            f"  Quantity Limit: {form.quantity_limit}\n"
        )

    # Clinical criteria section
    if state.clinical_criteria:
        crit = state.clinical_criteria
        sections.append(
            f"CLINICAL CRITERIA\n"
            f"  Criteria Met: {'Yes' if crit.criteria_met else 'No'}\n"
            f"  Source: {crit.guideline_source}\n"
            f"  Relevance Score: {crit.relevance_score:.2f}\n"
            f"  Guideline: {crit.guideline_text[:200]}...\n"
            if len(crit.guideline_text) > 200
            else f"CLINICAL CRITERIA\n"
            f"  Criteria Met: {'Yes' if crit.criteria_met else 'No'}\n"
            f"  Source: {crit.guideline_source}\n"
            f"  Guideline: {crit.guideline_text}\n"
        )

    # Interaction section
    if state.interactions and state.interactions.interactions:
        lines = [f"DRUG INTERACTIONS (max severity: {state.interactions.max_severity.value})"]
        for ix in state.interactions.interactions:
            lines.append(f"  • {ix.drug_pair}: {ix.severity.value} — {ix.description[:100]}")
        sections.append("\n".join(lines) + "\n")

    # Cost section
    if state.cost_estimate:
        cost = state.cost_estimate
        sections.append(
            f"COST ANALYSIS\n"
            f"  Plan Cost (30-day): ${cost.plan_cost_30day:,.2f}\n"
            f"  Member Cost: ${cost.member_copay:,.2f}\n"
            f"  Annual Estimate: ${cost.total_annual_cost:,.2f}\n"
        )
        if cost.cheaper_alternatives:
            alt_lines = ["  Alternatives:"]
            for alt in cost.cheaper_alternatives:
                alt_lines.append(
                    f"    - {alt['drug']}: ${alt['monthly_cost']:,.2f}/mo (Tier {alt['tier']})"
                )
            sections.append("\n".join(alt_lines) + "\n")

    # Decision rationale
    sections.append(
        f"RATIONALE\n"
        f"  {decision.rationale}\n"
    )

    if decision.cited_evidence:
        ev_lines = ["CITED EVIDENCE"]
        for ev in decision.cited_evidence:
            ev_lines.append(f"  • {ev}")
        sections.append("\n".join(ev_lines) + "\n")

    sections.append(
        f"RECOMMENDED ACTION\n"
        f"  {decision.recommended_action}\n"
    )

    sections.append(
        f"REVIEW METADATA\n"
        f"  Review Type: Automated\n"
        f"  Human Review Required: {'Yes' if decision.requires_human_review else 'No'}\n"
        f"  Confidence: {decision.confidence:.0%}\n"
        f"  Steps Completed: {len(state.review_steps)}\n"
        f"  Nodes: {', '.join(state.completed_nodes)}\n"
    )

    return "\n".join(sections)


def generate_llm_summary(
    state: AgentState,
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
) -> str:
    """Generate a natural language summary using GPT-4. More expensive but
    produces more readable output for provider notifications.

    Falls back to template mode if LLM call fails.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": 0.2,
        }
        if api_key:
            llm_kwargs["api_key"] = api_key

        llm = ChatOpenAI(**llm_kwargs)

        review_data = {
            "request": state.request.model_dump(),
            "eligibility": state.eligibility.model_dump() if state.eligibility else None,
            "formulary": state.formulary.model_dump() if state.formulary else None,
            "clinical_criteria": state.clinical_criteria.model_dump() if state.clinical_criteria else None,
            "interactions": state.interactions.model_dump() if state.interactions else None,
            "cost_estimate": state.cost_estimate.model_dump() if state.cost_estimate else None,
            "decision": state.decision.model_dump() if state.decision else None,
        }

        prompt = (
            "You are writing a prior authorization determination letter. "
            "Summarize the following review data into a clear, professional "
            "notification suitable for the prescribing provider. Include the "
            "decision, rationale with specific clinical evidence cited, and "
            "next steps. Be concise but complete.\n\n"
            f"Review Data:\n{review_data}"
        )

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        logger.warning("LLM summary failed, falling back to template: %s", e)
        return generate_template_summary(state)


def format_review_trace(review_steps: list[ReviewStep | dict]) -> str:
    """Format the review steps into a readable trace for the Gradio UI."""
    lines = []
    for i, step_data in enumerate(review_steps, 1):
        step = step_data if isinstance(step_data, ReviewStep) else ReviewStep(**step_data)
        lines.append(f"{'─' * 50}")
        lines.append(f"Step {i}: {step.step_name}")
        if step.tool_called:
            lines.append(f"  Tool: {step.tool_called}")
            if step.tool_input:
                lines.append(f"  Input: {step.tool_input}")
        lines.append(f"  Result: {step.reasoning}")
        lines.append("")
    return "\n".join(lines)
