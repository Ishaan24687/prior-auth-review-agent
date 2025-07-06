"""
Clinical guidelines RAG tool.

Uses ChromaDB as a vector store over simplified medical guidelines. In a real
system this would index the full text of UpToDate, MCG guidelines, or NCCN
pathways. For this demo we seed a small collection with excerpts covering the
most common PA scenarios.

# TODO: replace hardcoded guidelines with a proper ingestion pipeline
# TODO: add metadata filtering by guideline organization (ADA, ACCP, AAD, etc.)
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from langchain.tools import tool

from src.graph.state import ClinicalGuidelineResult

logger = logging.getLogger(__name__)

GUIDELINE_DOCUMENTS = [
    {
        "id": "ada-t2dm-glp1",
        "text": (
            "ADA Standards of Care 2024: For patients with type 2 diabetes mellitus (T2DM) "
            "who do not achieve glycemic targets on metformin monotherapy, GLP-1 receptor "
            "agonists (semaglutide, dulaglutide, liraglutide) are recommended as second-line "
            "therapy. GLP-1 RAs are preferred when the patient has established atherosclerotic "
            "cardiovascular disease (ASCVD), heart failure, or chronic kidney disease. HbA1c "
            "target is generally <7% for most adults."
        ),
        "source": "ADA Standards of Care 2024",
        "drug_class": "GLP-1 RA",
        "conditions": ["E11", "E11.65", "E11.9", "E11.21", "E11.22"],
    },
    {
        "id": "ada-t2dm-sglt2",
        "text": (
            "ADA Standards of Care 2024: SGLT2 inhibitors (empagliflozin, dapagliflozin, "
            "canagliflozin) are recommended as add-on to metformin for T2DM, particularly "
            "when heart failure or CKD risk reduction is a priority. Can be used as "
            "second-line or third-line therapy. Requires adequate renal function (eGFR "
            "considerations vary by agent)."
        ),
        "source": "ADA Standards of Care 2024",
        "drug_class": "SGLT2 inhibitor",
        "conditions": ["E11", "E11.65", "E11.9", "I50", "N18"],
    },
    {
        "id": "ada-t2dm-dpp4",
        "text": (
            "ADA Standards of Care 2024: DPP-4 inhibitors (sitagliptin, saxagliptin, "
            "linagliptin) may be used as second-line therapy for T2DM when GLP-1 RA is "
            "not appropriate. They are weight-neutral and have a favorable side-effect "
            "profile, but provide less HbA1c reduction than GLP-1 RAs. Not recommended "
            "for patients with heart failure (saxagliptin specifically contraindicated)."
        ),
        "source": "ADA Standards of Care 2024",
        "drug_class": "DPP-4 inhibitor",
        "conditions": ["E11", "E11.65", "E11.9"],
    },
    {
        "id": "accp-vte-doac",
        "text": (
            "ACCP 2024 Guidelines: For treatment of venous thromboembolism (VTE), direct "
            "oral anticoagulants (DOACs) — rivaroxaban, apixaban, edoxaban — are recommended "
            "over vitamin K antagonists (warfarin) for most patients. Treatment duration: "
            "minimum 3 months for provoked VTE, indefinite for unprovoked VTE with low "
            "bleeding risk. Apixaban and rivaroxaban do not require initial parenteral "
            "anticoagulation."
        ),
        "source": "ACCP VTE Guidelines 2024",
        "drug_class": "DOAC",
        "conditions": ["I26", "I26.99", "I82", "I82.40", "I82.90"],
    },
    {
        "id": "accp-afib-anticoag",
        "text": (
            "ACCP/AHA 2024: For patients with non-valvular atrial fibrillation and "
            "CHA2DS2-VASc score >= 2 (men) or >= 3 (women), anticoagulation is recommended "
            "to prevent stroke. DOACs are preferred over warfarin. Apixaban has the most "
            "favorable bleeding profile among DOACs."
        ),
        "source": "ACCP/AHA AF Guidelines 2024",
        "drug_class": "DOAC",
        "conditions": ["I48", "I48.0", "I48.1", "I48.91"],
    },
    {
        "id": "aad-psoriasis-biologic",
        "text": (
            "AAD-NPF Joint Guidelines 2024: Biologic therapy (TNF inhibitors: adalimumab, "
            "etanercept; IL-17 inhibitors: secukinumab; IL-23 inhibitors: risankizumab, "
            "ustekinumab) is recommended for moderate-to-severe plaque psoriasis after "
            "failure of conventional systemic therapy (methotrexate, cyclosporine, or "
            "phototherapy). BSA > 10% or DLQI > 10 qualifies as moderate-to-severe. "
            "Biologics may be first-line for severe disease with BSA > 20%."
        ),
        "source": "AAD-NPF Joint Guidelines 2024",
        "drug_class": "biologic",
        "conditions": ["L40", "L40.0", "L40.1", "L40.4"],
    },
    {
        "id": "acr-ra-biologic",
        "text": (
            "ACR 2024 Guidelines: For rheumatoid arthritis (RA) patients with moderate-to-high "
            "disease activity despite methotrexate (or other conventional DMARD), biologic "
            "DMARDs (adalimumab, etanercept, tocilizumab) are recommended. TNF inhibitors "
            "are first-line biologic. JAK inhibitors (tofacitinib) are an alternative but "
            "carry boxed warnings for cardiovascular and malignancy risk."
        ),
        "source": "ACR RA Guidelines 2024",
        "drug_class": "biologic DMARD",
        "conditions": ["M05", "M05.79", "M06", "M06.9"],
    },
    {
        "id": "nccn-melanoma-pembro",
        "text": (
            "NCCN Guidelines 2024: Pembrolizumab (anti-PD-1) is recommended as first-line "
            "therapy for unresectable or metastatic melanoma (Stage III/IV). Also indicated "
            "as adjuvant therapy for resected Stage IIB-IV melanoma. PD-L1 testing is "
            "recommended but not required for treatment initiation. Monitor for immune-related "
            "adverse events including colitis, hepatitis, and pneumonitis."
        ),
        "source": "NCCN Melanoma Guidelines 2024",
        "drug_class": "immune checkpoint inhibitor",
        "conditions": ["C43", "C43.9", "C43.4", "C43.5"],
    },
    {
        "id": "acc-hf-arni",
        "text": (
            "ACC/AHA Heart Failure Guidelines 2024: For HFrEF (LVEF <= 40%), sacubitril/"
            "valsartan (ARNI) is recommended as a replacement for ACE inhibitor or ARB "
            "to further reduce morbidity and mortality. Patients should be tolerating an "
            "ACE inhibitor or ARB before switching. Requires 36-hour washout period from "
            "ACE inhibitor before initiating ARNI."
        ),
        "source": "ACC/AHA HF Guidelines 2024",
        "drug_class": "ARNI",
        "conditions": ["I50", "I50.2", "I50.20", "I50.22"],
    },
    {
        "id": "aga-gerd",
        "text": (
            "AGA Clinical Practice Guidelines 2024: Proton pump inhibitors (PPIs) are "
            "recommended as first-line therapy for GERD with erosive esophagitis. For "
            "non-erosive GERD, a trial of PPI for 8 weeks is recommended. Long-term PPI "
            "use should be at the lowest effective dose. H2 receptor antagonists may be "
            "used as step-down therapy."
        ),
        "source": "AGA GERD Guidelines 2024",
        "drug_class": "PPI",
        "conditions": ["K21", "K21.0", "K21.9"],
    },
    {
        "id": "ada-weight-management",
        "text": (
            "ADA/Endocrine Society 2024: Anti-obesity medications (semaglutide 2.4mg [Wegovy], "
            "tirzepatide [Zepbound]) are recommended for adults with BMI >= 30 or BMI >= 27 "
            "with weight-related comorbidity, as adjunct to lifestyle intervention. These "
            "medications are NOT indicated solely for cosmetic weight loss. Diabetes and "
            "cardiovascular risk factor improvement should be documented."
        ),
        "source": "ADA/Endocrine Society Obesity Guidelines 2024",
        "drug_class": "anti-obesity",
        "conditions": ["E66", "E66.01", "E66.09"],
    },
    {
        "id": "neuropathic-pain",
        "text": (
            "AAN Neuropathic Pain Guidelines 2024: First-line therapy for neuropathic pain "
            "includes gabapentin, pregabalin, duloxetine, or tricyclic antidepressants. "
            "Pregabalin should be tried after gabapentin failure or intolerance. "
            "Combination therapy may be considered for refractory cases. Opioids are "
            "NOT recommended as first- or second-line therapy."
        ),
        "source": "AAN Neuropathic Pain Guidelines 2024",
        "drug_class": "neuropathic pain",
        "conditions": ["G89.29", "G62.9", "E11.42", "G63"],
    },
]

_chroma_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None


def _get_or_create_collection() -> chromadb.Collection:
    """Lazily initialize the ChromaDB client and seed the guidelines collection."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    _chroma_client = chromadb.Client()
    _collection = _chroma_client.get_or_create_collection(
        name="clinical_guidelines",
        metadata={"hnsw:space": "cosine"},
    )

    if _collection.count() == 0:
        _collection.add(
            ids=[g["id"] for g in GUIDELINE_DOCUMENTS],
            documents=[g["text"] for g in GUIDELINE_DOCUMENTS],
            metadatas=[
                {"source": g["source"], "drug_class": g["drug_class"]}
                for g in GUIDELINE_DOCUMENTS
            ],
        )
        logger.info(
            "Seeded clinical_guidelines collection with %d documents",
            len(GUIDELINE_DOCUMENTS),
        )

    return _collection


def _check_diagnosis_match(diagnosis_codes: list[str], guideline_conditions: list[str]) -> bool:
    """Check if any submitted diagnosis code matches guideline conditions.
    Supports prefix matching (e.g., E11 matches E11.65)."""
    for dx in diagnosis_codes:
        dx_clean = dx.strip().upper()
        for condition in guideline_conditions:
            if dx_clean.startswith(condition) or condition.startswith(dx_clean):
                return True
    return False


@tool
def clinical_guidelines(drug: str, diagnosis: str) -> str:
    """Search clinical guidelines for evidence supporting a drug for a given diagnosis.

    Args:
        drug: Drug name (brand or generic).
        diagnosis: Comma-separated ICD-10 codes or diagnosis description.
    """
    collection = _get_or_create_collection()

    query_text = f"{drug} for {diagnosis}"
    results = collection.query(
        query_texts=[query_text],
        n_results=3,
    )

    if not results["documents"] or not results["documents"][0]:
        return ClinicalGuidelineResult(
            error=f"No clinical guidelines found for {drug} + {diagnosis}",
        ).model_dump_json()

    best_doc = results["documents"][0][0]
    best_metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0] if results.get("distances") else 1.0
    relevance_score = max(0.0, 1.0 - distance)

    diagnosis_codes = [c.strip() for c in diagnosis.split(",")]
    guideline_entry = next(
        (g for g in GUIDELINE_DOCUMENTS if g["text"] == best_doc),
        None,
    )
    criteria_met = False
    if guideline_entry:
        criteria_met = _check_diagnosis_match(
            diagnosis_codes, guideline_entry["conditions"]
        )

    return ClinicalGuidelineResult(
        guideline_text=best_doc,
        criteria_met=criteria_met,
        guideline_source=best_metadata.get("source", ""),
        relevance_score=round(relevance_score, 3),
    ).model_dump_json()
