"""
Main ReAct clinical reviewer agent using LangChain.

This agent has access to all PA review tools and uses the clinical reviewer
system prompt with few-shot examples. It operates in a ReAct loop:
  Thought → Action → Observation → Thought → ... → Final Answer

The agent is used as an alternative to the deterministic graph workflow
for cases that need more flexible reasoning (e.g., off-label uses, complex
multi-drug interactions, appeals).

# TODO: add streaming support for real-time UI updates
# TODO: implement confidence calibration based on historical accuracy
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.graph.state import PARequest
from src.prompts.reviewer_prompt import REVIEWER_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from src.tools.clinical_guidelines import clinical_guidelines
from src.tools.cost_estimator import cost_estimator
from src.tools.drug_interaction import drug_interaction
from src.tools.eligibility_check import eligibility_check
from src.tools.formulary_lookup import formulary_lookup
from src.tools.icd10_validator import icd10_validator

logger = logging.getLogger(__name__)

PA_TOOLS = [
    eligibility_check,
    formulary_lookup,
    clinical_guidelines,
    drug_interaction,
    icd10_validator,
    cost_estimator,
]

REACT_TEMPLATE = """You are a clinical pharmacist reviewing a prior authorization request.

{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the prior authorization request to review
Thought: I need to think about what information I need to gather
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to make a decision
Final Answer: the PA decision with full rationale

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def create_reviewer_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    max_iterations: int = 10,
    api_key: Optional[str] = None,
) -> AgentExecutor:
    """Create and return a configured ReAct PA reviewer agent.

    Args:
        model_name: OpenAI model to use.
        temperature: Sampling temperature — kept low for consistency.
        max_iterations: Max ReAct loop iterations before forced termination.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
    }
    if api_key:
        llm_kwargs["api_key"] = api_key

    llm = ChatOpenAI(**llm_kwargs)

    prompt = PromptTemplate(
        template=REACT_TEMPLATE,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "system_prompt": REVIEWER_SYSTEM_PROMPT,
            "tools": "\n".join(
                f"  {t.name}: {t.description}" for t in PA_TOOLS
            ),
            "tool_names": ", ".join(t.name for t in PA_TOOLS),
        },
    )

    agent = create_react_agent(
        llm=llm,
        tools=PA_TOOLS,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=PA_TOOLS,
        max_iterations=max_iterations,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def review_pa_request(
    request: PARequest,
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Run the ReAct agent on a PA request and return structured results.

    Returns a dict with:
      - decision: the agent's final answer
      - intermediate_steps: list of (action, observation) tuples
      - trace: formatted string of the full reasoning chain
    """
    agent = create_reviewer_agent(model_name=model_name, api_key=api_key)

    input_text = (
        f"Prior Authorization Request:\n"
        f"  Member ID: {request.member_id}\n"
        f"  Drug: {request.drug_name}\n"
        f"  Diagnosis Codes: {', '.join(request.diagnosis_codes)}\n"
        f"  Current Medications: {', '.join(request.current_medications)}\n"
        f"  Provider NPI: {request.provider_npi}\n"
        f"  Quantity: {request.quantity}\n"
        f"  Days Supply: {request.days_supply}\n"
        f"  Urgency: {request.urgency.value}\n"
    )

    try:
        result = agent.invoke({"input": input_text})
    except Exception as e:
        logger.error("Agent execution failed: %s", e)
        return {
            "decision": f"ERROR: Agent execution failed — {e}",
            "intermediate_steps": [],
            "trace": f"Agent error: {e}",
        }

    intermediate_steps = result.get("intermediate_steps", [])
    trace_lines = []
    for i, (action, observation) in enumerate(intermediate_steps, 1):
        trace_lines.append(f"Step {i}:")
        trace_lines.append(f"  Action: {action.tool}")
        trace_lines.append(f"  Input: {action.tool_input}")
        obs_preview = str(observation)[:500]
        trace_lines.append(f"  Observation: {obs_preview}")
        trace_lines.append("")

    return {
        "decision": result.get("output", "No decision produced"),
        "intermediate_steps": intermediate_steps,
        "trace": "\n".join(trace_lines),
    }


def run_deterministic_review(request: PARequest) -> dict[str, Any]:
    """Run the deterministic LangGraph workflow (no LLM calls) for faster,
    cheaper review. Falls back to agent for ambiguous cases.

    This is the primary review path — the agent path above is for cases
    where the deterministic workflow pends or when we need the LLM's
    reasoning for complex clinical scenarios.
    """
    from src.graph.workflow import pa_review_chain

    initial_state = {
        "request": request.model_dump(),
        "eligibility": None,
        "formulary": None,
        "clinical_criteria": None,
        "interactions": None,
        "cost_estimate": None,
        "icd10_valid": True,
        "icd10_details": [],
        "review_steps": [],
        "decision": None,
        "summary": "",
        "error": None,
        "completed_nodes": [],
    }

    try:
        final_state = pa_review_chain.invoke(initial_state)
        return {
            "decision": final_state.get("decision", {}),
            "summary": final_state.get("summary", ""),
            "review_steps": final_state.get("review_steps", []),
            "completed_nodes": final_state.get("completed_nodes", []),
        }
    except Exception as e:
        logger.error("Deterministic workflow failed: %s", e)
        return {
            "decision": {"decision": "pend", "rationale": f"Workflow error: {e}"},
            "summary": f"ERROR: Workflow failed — {e}",
            "review_steps": [],
            "completed_nodes": [],
        }
