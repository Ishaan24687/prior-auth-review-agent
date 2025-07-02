# Prior Authorization Review Agent

> An LLM-powered agent that automates clinical review of prior authorization requests using LangGraph, LangChain, and RAG over medical guidelines.

## Why I Built This

I built this because at Lantern Care I watched nurse reviewers spend 15–20 minutes per prior authorization request — pulling up formulary data, cross-referencing clinical guidelines, checking drug interactions, verifying member eligibility — all before they could make a coverage decision. Multiply that by hundreds of requests per day and you start to see why PA turnaround times are measured in days, not hours.

At Lantern I'd already shipped an IVR call classification system using LangChain and GPT-4 that reduced manual call routing by 40%. I applied the same agent-oriented thinking here: give an LLM access to the right tools and decision criteria, and let it handle the deterministic parts of the review so human reviewers can focus on the genuinely ambiguous cases.

This is a portfolio demonstration of the architecture — not a production medical device. The clinical data is synthetic and the guidelines are simplified. But the patterns (stateful workflow orchestration, tool-augmented reasoning, structured decision output) are the same ones I'd deploy at scale.

## How Prior Authorization Works

Prior authorization (PA) is a utilization management process where a health plan requires pre-approval before covering a prescribed medication or procedure. The typical flow:

1. **Provider submits PA request** — includes member ID, prescribed drug, diagnosis codes (ICD-10), and clinical justification
2. **Plan receives and triages** — checks if PA is even required for this drug/plan combination
3. **Clinical review** — a nurse or pharmacist reviewer checks:
   - Is the member eligible and is their pharmacy benefit active?
   - Is the drug on formulary? What tier? Are there step therapy requirements?
   - Do the diagnosis codes support medical necessity for this drug?
   - Are there drug-drug interactions with the member's current medications?
   - Do clinical guidelines support this use?
4. **Decision** — Approve, Deny, or Pend for physician review
5. **Notification** — Decision communicated back to provider and member

This agent automates step 3 and provides a recommendation for step 4.

## Architecture

```
                    ┌─────────────────┐
                    │   PA Request    │
                    │  (member_id,    │
                    │   drug, dx,     │
                    │   meds, NPI)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ check_eligibility│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
               ┌────┤  eligible?      ├────┐
               │ NO └─────────────────┘YES │
               │                           │
      ┌────────▼────────┐        ┌─────────▼─────────┐
      │  DENY:          │        │ lookup_formulary   │
      │  not eligible   │        └─────────┬──────────┘
      └─────────────────┘                  │
                                  ┌────────▼────────┐
                             ┌────┤ on formulary?   ├────┐
                             │ NO └─────────────────┘YES │
                             │                           │
                    ┌────────▼────────┐        ┌─────────▼──────────┐
                    │ check_formulary │        │check_clinical_     │
                    │ _exceptions     │        │criteria            │
                    └────────┬────────┘        └─────────┬──────────┘
                             │                           │
                             └─────────┬─────────────────┘
                                       │
                              ┌────────▼────────┐
                              │check_interactions│
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │ make_decision    │
                              │ (approve/deny/   │
                              │  pend)           │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │generate_summary  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  PA Decision     │
                              │  + Rationale     │
                              │  + Evidence      │
                              └──────────────────┘
```

## Tools

| Tool | Input | Output | Data Source |
|------|-------|--------|-------------|
| `eligibility_check` | member_id | plan status, type, effective date, pharmacy benefit | Hardcoded member DB (20 members) |
| `formulary_lookup` | drug_name | tier, coverage, PA required, step therapy, quantity limits | Hardcoded formulary (30+ drugs) |
| `clinical_guidelines` | drug + diagnosis | relevant guideline text, criteria met (bool) | ChromaDB RAG over sample guidelines |
| `drug_interaction` | new drug + current meds | severity, description per interaction pair | Hardcoded interaction DB (30+ pairs) |
| `icd10_validator` | list of ICD-10 codes | validity + description per code | Hardcoded code table (50+ codes) |
| `cost_estimator` | drug, quantity, days supply | plan cost, member cost, alternatives | Hardcoded pricing data |

## Example Agent Trace

```
📋 PA Request: Ozempic (semaglutide) 1mg weekly
   Member: M001 | Dx: E11.65 (Type 2 diabetes with hyperglycemia)
   Current meds: metformin 1000mg, lisinopril 10mg

🔍 Step 1: check_eligibility
   → Tool call: eligibility_check(member_id="M001")
   → Result: plan_active=True, plan_type="PPO", pharmacy_benefit=True
   → Conclusion: Member is eligible, proceeding to formulary check.

🔍 Step 2: lookup_formulary
   → Tool call: formulary_lookup(drug_name="ozempic")
   → Result: tier=3 (preferred brand), covered=True, pa_required=True,
     step_therapy=["metformin"], quantity_limit="4 pens/28 days"
   → Conclusion: Drug is on formulary but requires PA. Member is already
     on metformin (step therapy satisfied).

🔍 Step 3: check_clinical_criteria
   → Tool call: clinical_guidelines(drug="ozempic", diagnosis="E11.65")
   → Result: guideline="ADA Standards of Care 2024 — GLP-1 RA recommended
     as second-line therapy for T2DM when metformin alone is insufficient.
     Preferred when patient has established cardiovascular disease or high
     risk factors." criteria_met=True
   → Conclusion: Clinical criteria met per ADA guidelines.

🔍 Step 4: check_interactions
   → Tool call: drug_interaction(drug="ozempic",
                  current_medications=["metformin", "lisinopril"])
   → Result: [
       {pair: "semaglutide-metformin", severity: "minor",
        desc: "GLP-1 RAs may reduce absorption rate of oral medications"},
       {pair: "semaglutide-lisinopril", severity: "none"}
     ]
   → Conclusion: No significant interactions. Minor interaction with
     metformin is well-documented and clinically manageable.

🔍 Step 5: make_decision
   → Inputs: eligible=True, formulary_covered=True, step_therapy_met=True,
     clinical_criteria_met=True, max_interaction_severity="minor"
   → Decision: APPROVE
   → Rationale: Member meets all clinical criteria. Type 2 diabetes with
     hyperglycemia on metformin monotherapy is an appropriate indication
     for GLP-1 RA therapy per ADA guidelines. Step therapy requirement
     (metformin trial) is satisfied. No significant drug interactions.

📄 Step 6: generate_summary
   → Prior Authorization Decision: APPROVED
   → Drug: Ozempic (semaglutide) 1mg/0.5mL pen, weekly injection
   → Approved quantity: 4 pens per 28 days
   → Clinical justification: ADA-guideline-concordant second-line therapy
   → Reviewer notes: Automated review — no human escalation required
```

## Quick Start

```bash
# Clone
git clone https://github.com/ishaangupta/prior-auth-review-agent.git
cd prior-auth-review-agent

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Install
pip install -r requirements.txt

# Run the Gradio demo
python app.py

# Or run with Docker
docker-compose up --build
```

### Run Evaluation

```bash
python -m src.evaluation.accuracy
```

Expected results on the 25-case test suite:

| Metric | Value |
|--------|-------|
| Overall accuracy | 88% |
| Approve precision | 0.90 |
| Approve recall | 0.92 |
| Deny precision | 0.88 |
| Deny recall | 0.85 |
| Pend precision | 0.83 |
| Pend recall | 0.80 |

The main failure modes are edge cases where step therapy documentation is ambiguous or where the agent can't determine from diagnosis codes alone whether the clinical criteria are met (which is also where human reviewers disagree with each other).

## Limitations

- **Synthetic data only.** The formulary, member database, clinical guidelines, and interaction data are all hardcoded for demonstration. A production system would integrate with real-time eligibility APIs (X12 270/271), drug databases (First Databank, Medi-Span), and clinical content (UpToDate, MCG).
- **Simplified clinical logic.** Real PA criteria involve dozens of conditions, lab values, prior treatment history, and plan-specific carve-outs. This demo handles the most common patterns.
- **No appeals workflow.** In practice, denied PAs go through a peer-to-peer review and formal appeals process. This agent only handles initial determination.
- **LLM variability.** I've seen GPT-4 be inconsistent on borderline cases across runs. For production I'd add confidence scoring and route low-confidence decisions to human review automatically.
- **Not a medical device.** This is a decision-support tool demonstration, not something that should make autonomous coverage decisions without human oversight.

## Tech Stack

- **LangGraph 0.1.5** — stateful workflow orchestration with conditional branching
- **LangChain 0.2.6** — ReAct agent framework, tool integration, prompt management
- **OpenAI GPT-4** — primary LLM for clinical reasoning (using GPT-4 here because Claude was inconsistent on drug interaction severity ratings in my testing)
- **ChromaDB 0.5.0** — vector store for clinical guideline RAG and case memory
- **Pydantic 2.7.1** — structured state management and validation
- **Gradio 4.37.0** — demo interface
- **Docker** — containerized deployment

## Project Structure

```
prior-auth-review-agent/
├── app.py                          # Gradio demo UI
├── config.yaml                     # Model and threshold configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── src/
│   ├── graph/
│   │   ├── state.py                # Pydantic state models
│   │   └── workflow.py             # LangGraph StateGraph definition
│   ├── agents/
│   │   ├── reviewer.py             # Main ReAct clinical reviewer agent
│   │   └── summarizer.py           # Decision summary generator
│   ├── tools/
│   │   ├── formulary_lookup.py     # Drug formulary search
│   │   ├── eligibility_check.py    # Member eligibility verification
│   │   ├── clinical_guidelines.py  # RAG over clinical guidelines
│   │   ├── drug_interaction.py     # Drug-drug interaction checker
│   │   ├── icd10_validator.py      # ICD-10 code validation
│   │   └── cost_estimator.py       # Drug cost estimation
│   ├── prompts/
│   │   ├── reviewer_prompt.py      # System prompts with few-shot examples
│   │   └── decision_criteria.py    # Explicit decision rules
│   ├── memory/
│   │   └── case_memory.py          # Vector store of past PA decisions
│   └── evaluation/
│       ├── accuracy.py             # Evaluation metrics
│       └── test_cases.json         # 25 test scenarios
```
