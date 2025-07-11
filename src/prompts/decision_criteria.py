"""
Explicit, deterministic decision rules for PA review.

The LLM handles nuance, but these threshold-based rules act as guardrails.
Any decision the LLM produces gets validated against these rules — if the
rule engine disagrees, we escalate to human review regardless.

This two-layer approach (LLM reasoning + rule engine validation) is how we
got the false-negative rate down to acceptable levels at Lantern.
"""

from src.graph.state import (
    AgentState,
    InteractionSeverity,
    PADecision,
    PADecisionType,
)

SEVERITY_WEIGHTS = {
    InteractionSeverity.NONE: 0.0,
    InteractionSeverity.MINOR: 0.1,
    InteractionSeverity.MODERATE: 0.4,
    InteractionSeverity.SEVERE: 1.0,
}

HIGH_COST_THRESHOLD_USD = 5000.0
AUTO_APPROVE_CONFIDENCE = 0.85
AUTO_DENY_CONFIDENCE = 0.90
PEND_THRESHOLD = 0.60


def evaluate_decision_rules(state: AgentState) -> PADecision:
    """Apply deterministic rules to the accumulated state and return a
    PA decision. This runs AFTER all tool calls have populated the state."""

    reasons: list[str] = []
    cited_evidence: list[str] = []
    deny_flags: list[str] = []
    pend_flags: list[str] = []

    # --- Rule 1: Eligibility gate ---
    if state.eligibility and not state.eligibility.plan_active:
        deny_flags.append("Member plan is not active")
        cited_evidence.append(
            f"Eligibility check: plan_active={state.eligibility.plan_active}"
        )

    if state.eligibility and not state.eligibility.pharmacy_benefit_active:
        deny_flags.append("Pharmacy benefit is not active")
        cited_evidence.append(
            f"Eligibility check: pharmacy_benefit_active="
            f"{state.eligibility.pharmacy_benefit_active}"
        )

    # --- Rule 2: Formulary coverage ---
    if state.formulary and not state.formulary.covered:
        deny_flags.append(
            f"{state.request.drug_name} is not covered on formulary"
        )
        cited_evidence.append(
            f"Formulary lookup: covered={state.formulary.covered}"
        )

    # --- Rule 3: Step therapy ---
    if state.formulary and state.formulary.step_therapy_required:
        step_drugs = {d.lower() for d in state.formulary.step_therapy_drugs}
        current_meds = {m.lower().split()[0] for m in state.request.current_medications}
        if not step_drugs.intersection(current_meds):
            deny_flags.append(
                f"Step therapy not met. Required: {state.formulary.step_therapy_drugs}. "
                f"Current meds: {state.request.current_medications}"
            )
            cited_evidence.append("Step therapy requirement not satisfied")

    # --- Rule 4: Clinical criteria ---
    if state.clinical_criteria and not state.clinical_criteria.criteria_met:
        pend_flags.append("Clinical criteria not clearly met")
        cited_evidence.append(
            f"Clinical guideline: criteria_met={state.clinical_criteria.criteria_met}, "
            f"source={state.clinical_criteria.guideline_source}"
        )

    # --- Rule 5: Drug interactions ---
    if state.interactions:
        max_sev = state.interactions.max_severity
        if max_sev == InteractionSeverity.SEVERE:
            deny_flags.append(
                f"Severe drug interaction detected: "
                f"{[i.drug_pair for i in state.interactions.interactions if i.severity == InteractionSeverity.SEVERE]}"
            )
            cited_evidence.append("Severe interaction — contraindicated")
        elif max_sev == InteractionSeverity.MODERATE:
            pend_flags.append(
                "Moderate drug interaction requires clinical judgment"
            )
            cited_evidence.append(
                f"Moderate interaction: "
                f"{[i.drug_pair for i in state.interactions.interactions if i.severity == InteractionSeverity.MODERATE]}"
            )

    # --- Rule 6: ICD-10 validity ---
    if not state.icd10_valid:
        pend_flags.append("One or more ICD-10 codes are invalid")
        cited_evidence.append(f"ICD-10 validation: {state.icd10_details}")

    # --- Rule 7: Cost threshold ---
    if state.cost_estimate and state.cost_estimate.plan_cost_30day > HIGH_COST_THRESHOLD_USD:
        pend_flags.append(
            f"High-cost medication: ${state.cost_estimate.plan_cost_30day:.2f}/month "
            f"exceeds ${HIGH_COST_THRESHOLD_USD:.2f} threshold"
        )
        cited_evidence.append("High-cost flag triggered")

    # --- Decision logic ---
    if deny_flags:
        decision_type = PADecisionType.DENY
        rationale = "DENIED. " + "; ".join(deny_flags)
        confidence = AUTO_DENY_CONFIDENCE
        requires_human = False
        recommended_action = (
            "Notify provider of denial reason. "
            "Provider may resubmit with additional documentation or request peer-to-peer review."
        )
    elif pend_flags:
        decision_type = PADecisionType.PEND
        rationale = "PENDED FOR REVIEW. " + "; ".join(pend_flags)
        confidence = PEND_THRESHOLD
        requires_human = True
        recommended_action = (
            "Route to clinical pharmacist for manual review. "
            f"Flags: {'; '.join(pend_flags)}"
        )
    else:
        decision_type = PADecisionType.APPROVE
        rationale = (
            f"All criteria met for {state.request.drug_name}. "
            f"Member eligible, drug on formulary, clinical guidelines support use, "
            f"no significant interactions."
        )
        confidence = AUTO_APPROVE_CONFIDENCE
        requires_human = False
        recommended_action = (
            f"Approve {state.request.drug_name} for {state.request.days_supply} days supply. "
            f"Standard renewal period."
        )

    reasons.extend(deny_flags + pend_flags)

    return PADecision(
        decision=decision_type,
        confidence=confidence,
        rationale=rationale,
        cited_evidence=cited_evidence,
        recommended_action=recommended_action,
        requires_human_review=requires_human,
    )
