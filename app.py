"""
Gradio demo for the prior authorization review agent.

Provides a web interface where users can submit PA requests and see the
agent's step-by-step reasoning, decision, and full review trace.

Run: python app.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from src.agents.reviewer import run_deterministic_review
from src.agents.summarizer import format_review_trace, generate_template_summary
from src.graph.state import AgentState, PARequest, ReviewStep, UrgencyLevel
from src.memory.case_memory import retrieve_similar_cases

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_MEMBERS = [
    "M001 — James Wilson (PPO, active)",
    "M002 — Sarah Chen (HMO, active)",
    "M003 — Robert Martinez (PPO, active)",
    "M004 — Emily Johnson (EPO, INACTIVE)",
    "M005 — Michael Brown (HMO, active)",
    "M006 — Jessica Lee (PPO, no Rx benefit)",
    "M007 — David Kim (HMO, active)",
    "M008 — Amanda Rivera (PPO, active)",
    "M012 — Jennifer Taylor (PPO, active)",
    "M013 — Daniel Anderson (HMO, active)",
]

SAMPLE_DRUGS = [
    "ozempic", "humira", "eliquis", "xarelto", "jardiance", "entresto",
    "keytruda", "pregabalin", "mounjaro", "trulicity", "januvia",
    "farxiga", "wegovy", "enbrel", "stelara", "revlimid", "duloxetine",
    "atorvastatin", "metformin", "warfarin", "rosuvastatin",
]

COMMON_DIAGNOSIS_CODES = {
    "E11.65 — Type 2 diabetes with hyperglycemia": "E11.65",
    "E11.9 — Type 2 diabetes, uncomplicated": "E11.9",
    "E11.22 — T2DM with diabetic CKD": "E11.22",
    "E11.42 — T2DM with diabetic polyneuropathy": "E11.42",
    "I48.91 — Atrial fibrillation, unspecified": "I48.91",
    "I26.99 — Pulmonary embolism": "I26.99",
    "I50.22 — Chronic systolic heart failure": "I50.22",
    "L40.0 — Psoriasis vulgaris": "L40.0",
    "M06.9 — Rheumatoid arthritis, unspecified": "M06.9",
    "C43.9 — Malignant melanoma of skin": "C43.9",
    "C90.00 — Multiple myeloma": "C90.00",
    "E78.00 — Hypercholesterolemia": "E78.00",
    "E66.01 — Morbid obesity": "E66.01",
    "G89.29 — Chronic pain": "G89.29",
}


def process_pa_request(
    member_selection: str,
    drug_name: str,
    diagnosis_selections: list[str],
    current_meds: str,
    quantity: int,
    days_supply: int,
) -> tuple[str, str, str, str]:
    """Process a PA request through the deterministic workflow and return
    the decision, summary, review trace, and similar cases."""

    member_id = member_selection.split(" ")[0] if member_selection else "M001"

    dx_codes = []
    for sel in (diagnosis_selections or []):
        code = COMMON_DIAGNOSIS_CODES.get(sel, sel)
        dx_codes.append(code)
    if not dx_codes:
        dx_codes = ["E11.9"]

    med_list = [m.strip() for m in current_meds.split(",") if m.strip()] if current_meds else []

    request = PARequest(
        member_id=member_id,
        drug_name=drug_name or "metformin",
        diagnosis_codes=dx_codes,
        current_medications=med_list,
        provider_npi="0000000000",
        quantity=quantity or 30,
        days_supply=days_supply or 30,
        urgency=UrgencyLevel.STANDARD,
    )

    try:
        result = run_deterministic_review(request)
    except Exception as e:
        error_msg = f"Workflow failed: {e}"
        return error_msg, error_msg, error_msg, ""

    decision_data = result.get("decision", {})
    if isinstance(decision_data, dict):
        decision_type = decision_data.get("decision", "unknown").upper()
        confidence = decision_data.get("confidence", 0.0)
        rationale = decision_data.get("rationale", "")
        decision_display = (
            f"## Decision: {decision_type}\n\n"
            f"**Confidence:** {confidence:.0%}\n\n"
            f"**Rationale:** {rationale}\n\n"
            f"**Recommended Action:** {decision_data.get('recommended_action', '')}\n\n"
            f"**Human Review Required:** {'Yes' if decision_data.get('requires_human_review') else 'No'}"
        )
    else:
        decision_display = f"## Decision\n\n{decision_data}"

    summary = result.get("summary", "No summary generated.")

    review_steps = result.get("review_steps", [])
    trace = format_review_trace(review_steps)

    try:
        similar = retrieve_similar_cases(
            drug=request.drug_name,
            diagnosis=",".join(request.diagnosis_codes),
            n_results=3,
        )
        case_lines = []
        for case in similar:
            case_lines.append(
                f"**{case['id']}** — {case['drug']} for {case['diagnosis']}\n"
                f"  Decision: {case['decision'].upper()} (similarity: {case['similarity_score']:.2f})\n"
                f"  {case['rationale'][:150]}\n"
            )
        similar_display = "\n".join(case_lines) if case_lines else "No similar cases found."
    except Exception:
        similar_display = "Case memory unavailable."

    return decision_display, summary, trace, similar_display


def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="Prior Auth Review Agent",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="gray",
        ),
    ) as demo:
        gr.Markdown(
            "# Prior Authorization Review Agent\n"
            "Submit a PA request and see the agent's step-by-step clinical review.\n"
            "Uses a LangGraph workflow with deterministic decision rules — no LLM API calls required for the demo."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### PA Request")
                member_dropdown = gr.Dropdown(
                    choices=SAMPLE_MEMBERS,
                    value=SAMPLE_MEMBERS[0],
                    label="Member",
                )
                drug_dropdown = gr.Dropdown(
                    choices=SAMPLE_DRUGS,
                    value="ozempic",
                    label="Drug",
                    allow_custom_value=True,
                )
                diagnosis_check = gr.CheckboxGroup(
                    choices=list(COMMON_DIAGNOSIS_CODES.keys()),
                    value=["E11.65 — Type 2 diabetes with hyperglycemia"],
                    label="Diagnosis Codes",
                )
                current_meds_input = gr.Textbox(
                    value="metformin 1000mg, lisinopril 10mg",
                    label="Current Medications (comma-separated)",
                    placeholder="e.g., metformin 1000mg, lisinopril 10mg",
                )
                with gr.Row():
                    quantity_input = gr.Number(value=4, label="Quantity", precision=0)
                    days_input = gr.Number(value=28, label="Days Supply", precision=0)

                submit_btn = gr.Button("Submit PA Review", variant="primary")

            with gr.Column(scale=2):
                decision_output = gr.Markdown(label="Decision")
                with gr.Accordion("Full Review Summary", open=False):
                    summary_output = gr.Textbox(
                        label="Summary",
                        lines=20,
                        interactive=False,
                    )
                with gr.Accordion("Agent Reasoning Trace", open=False):
                    trace_output = gr.Textbox(
                        label="Step-by-Step Trace",
                        lines=15,
                        interactive=False,
                    )
                with gr.Accordion("Similar Past Cases", open=False):
                    similar_output = gr.Markdown(label="Similar Cases")

        submit_btn.click(
            fn=process_pa_request,
            inputs=[
                member_dropdown,
                drug_dropdown,
                diagnosis_check,
                current_meds_input,
                quantity_input,
                days_input,
            ],
            outputs=[decision_output, summary_output, trace_output, similar_output],
        )

        gr.Markdown(
            "---\n"
            "*Built by Ishaan Gupta. This is a portfolio demo with synthetic data — "
            "not a production clinical decision support system.*"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
