"""
Vector store of past PA decisions for case-based reasoning.

When the agent encounters a new PA request, it retrieves similar past cases
to inform its decision. This is analogous to how human reviewers build
institutional knowledge over time — "I saw a similar Humira case last week
where we denied because step therapy wasn't met."

Uses ChromaDB with cosine similarity on a text representation of each case.

# TODO: add feedback loop — when human reviewers overturn an agent decision,
#       store the correction so the retrieval improves over time
# TODO: weight recent cases higher than older ones
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)

SEED_CASES: list[dict[str, Any]] = [
    {
        "id": "CASE-001",
        "drug": "ozempic",
        "diagnosis": "E11.65",
        "decision": "approve",
        "rationale": "T2DM with hyperglycemia, on metformin (step therapy met), ADA criteria met, no significant interactions.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-002",
        "drug": "humira",
        "diagnosis": "L40.0",
        "decision": "deny",
        "rationale": "Psoriasis vulgaris but no documentation of prior methotrexate or phototherapy trial. Step therapy not met.",
        "member_plan": "HMO",
    },
    {
        "id": "CASE-003",
        "drug": "xarelto",
        "diagnosis": "I26.99",
        "decision": "pend",
        "rationale": "PE diagnosis appropriate for DOAC, but moderate interaction with amiodarone requires pharmacist review.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-004",
        "drug": "eliquis",
        "diagnosis": "I48.91",
        "decision": "approve",
        "rationale": "Atrial fibrillation, DOAC preferred over warfarin per ACCP guidelines, no significant interactions.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-005",
        "drug": "jardiance",
        "diagnosis": "E11.65",
        "decision": "approve",
        "rationale": "T2DM on metformin (step therapy met), SGLT2i appropriate per ADA guidelines for CV risk reduction.",
        "member_plan": "HMO",
    },
    {
        "id": "CASE-006",
        "drug": "keytruda",
        "diagnosis": "C43.9",
        "decision": "pend",
        "rationale": "Melanoma dx supports pembrolizumab per NCCN, but high-cost specialty drug exceeds threshold. Requires medical director review.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-007",
        "drug": "humira",
        "diagnosis": "M06.9",
        "decision": "approve",
        "rationale": "RA with documented methotrexate failure. TNF inhibitor appropriate per ACR guidelines. Step therapy satisfied.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-008",
        "drug": "ozempic",
        "diagnosis": "E66.01",
        "decision": "deny",
        "rationale": "Submitted for obesity (E66.01) but Ozempic is only covered for T2DM indication. Wegovy (semaglutide for weight) is excluded from formulary.",
        "member_plan": "HMO",
    },
    {
        "id": "CASE-009",
        "drug": "entresto",
        "diagnosis": "I50.22",
        "decision": "approve",
        "rationale": "Chronic systolic HF, member on lisinopril (ACE inhibitor trial met), ARNI indicated per ACC/AHA guidelines.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-010",
        "drug": "pregabalin",
        "diagnosis": "G89.29",
        "decision": "deny",
        "rationale": "Chronic pain, but no documentation of gabapentin trial. Step therapy requires gabapentin before pregabalin.",
        "member_plan": "EPO",
    },
    {
        "id": "CASE-011",
        "drug": "wegovy",
        "diagnosis": "E66.01",
        "decision": "deny",
        "rationale": "Wegovy is excluded from formulary. Weight management medications not covered under current plan design.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-012",
        "drug": "mounjaro",
        "diagnosis": "E11.65",
        "decision": "deny",
        "rationale": "T2DM indication appropriate but step therapy requires metformin + ozempic trial before mounjaro. Only metformin documented.",
        "member_plan": "HMO",
    },
    {
        "id": "CASE-013",
        "drug": "enbrel",
        "diagnosis": "L40.0",
        "decision": "pend",
        "rationale": "Psoriasis with documented methotrexate intolerance (hepatotoxicity). Exception to step therapy may be warranted but needs medical director review.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-014",
        "drug": "duloxetine",
        "diagnosis": "E11.42",
        "decision": "approve",
        "rationale": "Diabetic polyneuropathy — duloxetine is first-line per AAN guidelines. Step therapy for depression indication not applicable here.",
        "member_plan": "PPO",
    },
    {
        "id": "CASE-015",
        "drug": "revlimid",
        "diagnosis": "C90.00",
        "decision": "pend",
        "rationale": "Multiple myeloma supports lenalidomide use per NCCN. High-cost specialty drug requires additional documentation and medical director sign-off.",
        "member_plan": "PPO",
    },
]

_case_collection: Optional[chromadb.Collection] = None


def _get_case_collection() -> chromadb.Collection:
    """Lazily initialize the case memory collection and seed with historical cases."""
    global _case_collection

    if _case_collection is not None:
        return _case_collection

    client = chromadb.Client()
    _case_collection = client.get_or_create_collection(
        name="pa_case_memory",
        metadata={"hnsw:space": "cosine"},
    )

    if _case_collection.count() == 0:
        _case_collection.add(
            ids=[c["id"] for c in SEED_CASES],
            documents=[
                f"Drug: {c['drug']}, Diagnosis: {c['diagnosis']}, "
                f"Decision: {c['decision']}, Rationale: {c['rationale']}"
                for c in SEED_CASES
            ],
            metadatas=[
                {
                    "drug": c["drug"],
                    "diagnosis": c["diagnosis"],
                    "decision": c["decision"],
                    "plan_type": c["member_plan"],
                }
                for c in SEED_CASES
            ],
        )
        logger.info("Seeded case memory with %d historical cases", len(SEED_CASES))

    return _case_collection


def retrieve_similar_cases(
    drug: str,
    diagnosis: str,
    n_results: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve similar past PA decisions based on drug and diagnosis.

    Returns a list of dicts with keys: id, drug, diagnosis, decision,
    rationale, similarity_score.
    """
    collection = _get_case_collection()

    query_text = f"Drug: {drug}, Diagnosis: {diagnosis}"
    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, collection.count()),
    )

    cases = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            similarity = max(0.0, 1.0 - distance)

            cases.append({
                "id": results["ids"][0][i],
                "drug": metadata.get("drug", ""),
                "diagnosis": metadata.get("diagnosis", ""),
                "decision": metadata.get("decision", ""),
                "rationale": doc,
                "similarity_score": round(similarity, 3),
            })

    return cases


def store_case(
    case_id: str,
    drug: str,
    diagnosis: str,
    decision: str,
    rationale: str,
    plan_type: str = "",
) -> None:
    """Store a new PA decision in the case memory for future retrieval."""
    collection = _get_case_collection()

    collection.add(
        ids=[case_id],
        documents=[
            f"Drug: {drug}, Diagnosis: {diagnosis}, "
            f"Decision: {decision}, Rationale: {rationale}"
        ],
        metadatas=[
            {
                "drug": drug,
                "diagnosis": diagnosis,
                "decision": decision,
                "plan_type": plan_type,
            }
        ],
    )
    logger.info("Stored case %s in case memory", case_id)
