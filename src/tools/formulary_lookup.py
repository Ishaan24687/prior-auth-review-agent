"""
Drug formulary lookup tool.

Hardcoded formulary with 30+ drugs modeled on a typical commercial PBM
formulary. Tier structure follows CMS standard:
  Tier 1: preferred generic
  Tier 2: non-preferred generic
  Tier 3: preferred brand
  Tier 4: non-preferred brand
  Tier 5: specialty

# TODO: integrate with real formulary API (FDB MedKnowledge or Medi-Span)
# TODO: add NDC-level lookups instead of just drug name matching
"""

from langchain.tools import tool

from src.graph.state import FormularyResult

FORMULARY_DB: dict[str, dict] = {
    "metformin": {
        "generic_name": "metformin HCl",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "180 tablets/30 days",
        "ndc_code": "00591-0386-01",
    },
    "atorvastatin": {
        "generic_name": "atorvastatin calcium",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00378-2080-77",
    },
    "lisinopril": {
        "generic_name": "lisinopril",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00093-7112-06",
    },
    "amlodipine": {
        "generic_name": "amlodipine besylate",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00093-3171-56",
    },
    "omeprazole": {
        "generic_name": "omeprazole",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 capsules/30 days",
        "ndc_code": "62175-0256-37",
    },
    "gabapentin": {
        "generic_name": "gabapentin",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "270 capsules/30 days",
        "ndc_code": "27241-0069-03",
    },
    "sertraline": {
        "generic_name": "sertraline HCl",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00093-7199-56",
    },
    "losartan": {
        "generic_name": "losartan potassium",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00093-7367-56",
    },
    "levothyroxine": {
        "generic_name": "levothyroxine sodium",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00378-1810-01",
    },
    "montelukast": {
        "generic_name": "montelukast sodium",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00093-7620-56",
    },
    "clopidogrel": {
        "generic_name": "clopidogrel bisulfate",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "63629-4021-01",
    },
    "rosuvastatin": {
        "generic_name": "rosuvastatin calcium",
        "tier": 2, "tier_label": "non-preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": True, "step_therapy_drugs": ["atorvastatin"],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00591-3744-30",
    },
    "pregabalin": {
        "generic_name": "pregabalin",
        "tier": 2, "tier_label": "non-preferred generic",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["gabapentin"],
        "quantity_limit": "60 capsules/30 days",
        "ndc_code": "00093-3219-56",
    },
    "duloxetine": {
        "generic_name": "duloxetine HCl",
        "tier": 2, "tier_label": "non-preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": True, "step_therapy_drugs": ["sertraline", "fluoxetine"],
        "quantity_limit": "30 capsules/30 days",
        "ndc_code": "00002-3240-30",
    },
    "jardiance": {
        "generic_name": "empagliflozin",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["metformin"],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00597-0152-30",
    },
    "ozempic": {
        "generic_name": "semaglutide",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["metformin"],
        "quantity_limit": "4 pens/28 days",
        "ndc_code": "00169-4132-12",
    },
    "eliquis": {
        "generic_name": "apixaban",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "60 tablets/30 days",
        "ndc_code": "00003-0894-21",
    },
    "xarelto": {
        "generic_name": "rivaroxaban",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "50458-0580-30",
    },
    "entresto": {
        "generic_name": "sacubitril/valsartan",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["lisinopril", "enalapril", "ramipril"],
        "quantity_limit": "60 tablets/30 days",
        "ndc_code": "00078-0696-15",
    },
    "trulicity": {
        "generic_name": "dulaglutide",
        "tier": 4, "tier_label": "non-preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["metformin", "ozempic"],
        "quantity_limit": "4 pens/28 days",
        "ndc_code": "00002-1506-80",
    },
    "januvia": {
        "generic_name": "sitagliptin",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["metformin"],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00006-0277-31",
    },
    "farxiga": {
        "generic_name": "dapagliflozin",
        "tier": 3, "tier_label": "preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["metformin"],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00310-6205-30",
    },
    "keytruda": {
        "generic_name": "pembrolizumab",
        "tier": 5, "tier_label": "specialty",
        "covered": True, "pa_required": True,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "200mg/3 weeks (provider-administered)",
        "ndc_code": "00006-3026-02",
    },
    "humira": {
        "generic_name": "adalimumab",
        "tier": 5, "tier_label": "specialty",
        "covered": True, "pa_required": True,
        "step_therapy_required": True,
        "step_therapy_drugs": ["methotrexate", "sulfasalazine", "hydroxychloroquine"],
        "quantity_limit": "2 pens/28 days",
        "ndc_code": "00074-4339-02",
    },
    "enbrel": {
        "generic_name": "etanercept",
        "tier": 5, "tier_label": "specialty",
        "covered": True, "pa_required": True,
        "step_therapy_required": True,
        "step_therapy_drugs": ["methotrexate", "sulfasalazine"],
        "quantity_limit": "4 syringes/28 days",
        "ndc_code": "58406-0425-04",
    },
    "stelara": {
        "generic_name": "ustekinumab",
        "tier": 5, "tier_label": "specialty",
        "covered": True, "pa_required": True,
        "step_therapy_required": True,
        "step_therapy_drugs": ["humira", "enbrel"],
        "quantity_limit": "1 syringe/84 days",
        "ndc_code": "57894-0060-03",
    },
    "revlimid": {
        "generic_name": "lenalidomide",
        "tier": 5, "tier_label": "specialty",
        "covered": True, "pa_required": True,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "21 capsules/28 days",
        "ndc_code": "59572-0410-00",
    },
    "wegovy": {
        "generic_name": "semaglutide (weight management)",
        "tier": 5, "tier_label": "specialty",
        "covered": False, "pa_required": True,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "N/A — not covered",
        "ndc_code": "00169-4601-11",
    },
    "mounjaro": {
        "generic_name": "tirzepatide",
        "tier": 4, "tier_label": "non-preferred brand",
        "covered": True, "pa_required": True,
        "step_therapy_required": True, "step_therapy_drugs": ["metformin", "ozempic"],
        "quantity_limit": "4 pens/28 days",
        "ndc_code": "00002-1524-80",
    },
    "skyrizi": {
        "generic_name": "risankizumab",
        "tier": 5, "tier_label": "specialty",
        "covered": True, "pa_required": True,
        "step_therapy_required": True,
        "step_therapy_drugs": ["humira", "enbrel"],
        "quantity_limit": "1 pen/84 days",
        "ndc_code": "00074-2100-01",
    },
    "warfarin": {
        "generic_name": "warfarin sodium",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00555-0970-02",
    },
    "amiodarone": {
        "generic_name": "amiodarone HCl",
        "tier": 1, "tier_label": "preferred generic",
        "covered": True, "pa_required": False,
        "step_therapy_required": False, "step_therapy_drugs": [],
        "quantity_limit": "30 tablets/30 days",
        "ndc_code": "00093-7237-01",
    },
}


@tool
def formulary_lookup(drug_name: str) -> str:
    """Look up a drug in the plan formulary. Returns tier, coverage status,
    prior auth requirements, step therapy requirements, and quantity limits.

    Args:
        drug_name: Name of the drug to look up (brand or generic).
    """
    normalized = drug_name.strip().lower()

    # direct match
    if normalized in FORMULARY_DB:
        entry = FORMULARY_DB[normalized]
        result = FormularyResult(drug_name=drug_name, **entry)
        return result.model_dump_json()

    # try matching on generic name
    for brand, entry in FORMULARY_DB.items():
        if normalized in entry["generic_name"].lower():
            result = FormularyResult(drug_name=brand, **entry)
            return result.model_dump_json()

    return FormularyResult(
        drug_name=drug_name,
        covered=False,
        error=f"Drug '{drug_name}' not found in formulary. May require manual lookup.",
    ).model_dump_json()
