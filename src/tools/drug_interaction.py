"""
Drug-drug interaction checker.

Hardcoded interaction database with 30+ clinically relevant pairs. Severity
levels follow FDA/First Databank classification:
  - none: no known interaction
  - minor: clinical effects unlikely, no action needed
  - moderate: may require monitoring or dose adjustment
  - severe: combination contraindicated or requires significant intervention

# TODO: integrate First Databank Drug Interaction API for production
# using GPT-4 here because Claude was inconsistent on drug interaction severity ratings
"""

from langchain.tools import tool

from src.graph.state import DrugInteraction, InteractionResult, InteractionSeverity

INTERACTION_DB: dict[tuple[str, str], dict] = {
    ("warfarin", "aspirin"): {
        "severity": "severe",
        "description": (
            "Concurrent use increases bleeding risk significantly. Both agents "
            "impair hemostasis through different mechanisms. If combination is "
            "necessary, use lowest effective aspirin dose and monitor INR closely."
        ),
    },
    ("warfarin", "amiodarone"): {
        "severity": "severe",
        "description": (
            "Amiodarone inhibits CYP2C9 and CYP3A4, significantly increasing "
            "warfarin levels. Warfarin dose should be reduced by 30-50% when "
            "initiating amiodarone. Monitor INR weekly for first month."
        ),
    },
    ("warfarin", "metronidazole"): {
        "severity": "severe",
        "description": (
            "Metronidazole inhibits CYP2C9, increasing warfarin effect and "
            "bleeding risk. Requires INR monitoring and potential warfarin "
            "dose reduction during concurrent use."
        ),
    },
    ("rivaroxaban", "amiodarone"): {
        "severity": "moderate",
        "description": (
            "Amiodarone is a P-glycoprotein inhibitor that may increase "
            "rivaroxaban plasma levels by 30-50%, elevating bleeding risk. "
            "Clinical monitoring recommended; dose adjustment may be warranted."
        ),
    },
    ("apixaban", "amiodarone"): {
        "severity": "moderate",
        "description": (
            "Amiodarone inhibits P-gp and CYP3A4, potentially increasing "
            "apixaban levels. Monitor for signs of bleeding. Generally considered "
            "a manageable interaction with clinical oversight."
        ),
    },
    ("metformin", "contrast_dye"): {
        "severity": "severe",
        "description": (
            "Iodinated contrast media can cause acute kidney injury, leading "
            "to metformin accumulation and lactic acidosis. Hold metformin "
            "48 hours before and after contrast administration."
        ),
    },
    ("semaglutide", "metformin"): {
        "severity": "minor",
        "description": (
            "GLP-1 RAs slow gastric emptying, which may reduce the absorption "
            "rate of oral medications including metformin. Clinical significance "
            "is minimal; no dose adjustment required."
        ),
    },
    ("semaglutide", "insulin"): {
        "severity": "moderate",
        "description": (
            "Concurrent GLP-1 RA and insulin increases hypoglycemia risk. "
            "Consider reducing insulin dose by 20% when initiating semaglutide. "
            "Increased glucose monitoring recommended during titration."
        ),
    },
    ("semaglutide", "lisinopril"): {
        "severity": "none",
        "description": "No clinically significant interaction identified.",
    },
    ("semaglutide", "atorvastatin"): {
        "severity": "none",
        "description": "No clinically significant interaction identified.",
    },
    ("empagliflozin", "metformin"): {
        "severity": "none",
        "description": (
            "Combination is commonly used and well-studied. No pharmacokinetic "
            "interaction. Fixed-dose combination products available."
        ),
    },
    ("empagliflozin", "insulin"): {
        "severity": "moderate",
        "description": (
            "Additive hypoglycemic effect. Consider reducing insulin dose when "
            "adding SGLT2 inhibitor. Monitor blood glucose closely during "
            "first 2-4 weeks."
        ),
    },
    ("adalimumab", "methotrexate"): {
        "severity": "minor",
        "description": (
            "Combination is standard of care for RA. Methotrexate may reduce "
            "immunogenicity of adalimumab. No dose adjustment needed."
        ),
    },
    ("adalimumab", "live_vaccine"): {
        "severity": "severe",
        "description": (
            "TNF inhibitors suppress immune response. Live vaccines are "
            "contraindicated during adalimumab therapy due to risk of "
            "disseminated infection. Wait 3 months after discontinuation."
        ),
    },
    ("atorvastatin", "gemfibrozil"): {
        "severity": "severe",
        "description": (
            "Gemfibrozil inhibits statin metabolism via OATP1B1 and CYP2C8, "
            "significantly increasing statin levels and rhabdomyolysis risk. "
            "Use fenofibrate instead if fibrate therapy is needed."
        ),
    },
    ("atorvastatin", "amlodipine"): {
        "severity": "minor",
        "description": (
            "Amlodipine may modestly increase atorvastatin levels via CYP3A4 "
            "inhibition. Limit atorvastatin to 20mg daily when co-administered "
            "with amlodipine per FDA labeling."
        ),
    },
    ("atorvastatin", "grapefruit"): {
        "severity": "minor",
        "description": (
            "Grapefruit juice inhibits CYP3A4, modestly increasing statin "
            "levels. Avoid large quantities of grapefruit juice."
        ),
    },
    ("lisinopril", "potassium"): {
        "severity": "moderate",
        "description": (
            "ACE inhibitors reduce aldosterone, increasing potassium retention. "
            "Concurrent potassium supplementation may cause hyperkalemia. "
            "Monitor serum potassium levels."
        ),
    },
    ("lisinopril", "losartan"): {
        "severity": "severe",
        "description": (
            "Dual RAAS blockade (ACE inhibitor + ARB) increases risk of "
            "hyperkalemia, hypotension, and renal impairment. Combination "
            "is generally not recommended (ONTARGET trial)."
        ),
    },
    ("lisinopril", "metformin"): {
        "severity": "none",
        "description": "No clinically significant interaction identified.",
    },
    ("gabapentin", "opioids"): {
        "severity": "severe",
        "description": (
            "Concurrent use of gabapentinoids and opioids increases risk of "
            "respiratory depression, sedation, and death. FDA boxed warning. "
            "Avoid combination when possible; if necessary, use lowest doses."
        ),
    },
    ("gabapentin", "antacids"): {
        "severity": "minor",
        "description": (
            "Aluminum/magnesium antacids reduce gabapentin bioavailability by "
            "~20%. Administer gabapentin at least 2 hours after antacids."
        ),
    },
    ("pregabalin", "opioids"): {
        "severity": "severe",
        "description": (
            "Same class-wide risk as gabapentin + opioids. FDA boxed warning "
            "for respiratory depression. Avoid combination."
        ),
    },
    ("sertraline", "tramadol"): {
        "severity": "severe",
        "description": (
            "Both agents increase serotonin. Risk of serotonin syndrome — "
            "a potentially life-threatening condition. Combination should be "
            "avoided. If necessary, use lowest doses and monitor closely."
        ),
    },
    ("sertraline", "nsaids"): {
        "severity": "moderate",
        "description": (
            "SSRIs impair platelet function. Concurrent NSAID use increases "
            "GI bleeding risk. Consider GI prophylaxis with PPI if combination "
            "is necessary."
        ),
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "moderate",
        "description": (
            "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel "
            "to its active metabolite. May reduce antiplatelet effect. Use "
            "pantoprazole as alternative PPI."
        ),
    },
    ("duloxetine", "tramadol"): {
        "severity": "severe",
        "description": (
            "Both increase serotonin levels. Risk of serotonin syndrome. "
            "Avoid concurrent use."
        ),
    },
    ("duloxetine", "lisinopril"): {
        "severity": "none",
        "description": "No clinically significant interaction identified.",
    },
    ("pembrolizumab", "corticosteroids"): {
        "severity": "moderate",
        "description": (
            "Systemic corticosteroids may reduce efficacy of immune checkpoint "
            "inhibitors. Baseline corticosteroid use (>10mg prednisone equivalent) "
            "at treatment initiation associated with worse outcomes. Corticosteroids "
            "for irAE management are acceptable."
        ),
    },
    ("sacubitril_valsartan", "lisinopril"): {
        "severity": "severe",
        "description": (
            "Concurrent ARNI and ACE inhibitor is contraindicated due to "
            "increased risk of angioedema. Requires 36-hour washout from "
            "ACE inhibitor before starting sacubitril/valsartan."
        ),
    },
    ("metoprolol", "verapamil"): {
        "severity": "severe",
        "description": (
            "Concurrent beta-blocker and non-dihydropyridine calcium channel "
            "blocker can cause severe bradycardia, heart block, and hypotension. "
            "Combination generally contraindicated."
        ),
    },
    ("amiodarone", "metoprolol"): {
        "severity": "moderate",
        "description": (
            "Additive AV nodal blockade. Amiodarone may increase metoprolol "
            "levels via CYP2D6 inhibition. Monitor heart rate and ECG."
        ),
    },
}


def _normalize_drug_name(name: str) -> str:
    """Normalize drug name for lookup. Handles common brand→generic mappings
    needed for interaction checking."""
    mapping = {
        "ozempic": "semaglutide",
        "rybelsus": "semaglutide",
        "wegovy": "semaglutide",
        "jardiance": "empagliflozin",
        "farxiga": "dapagliflozin",
        "humira": "adalimumab",
        "enbrel": "etanercept",
        "eliquis": "apixaban",
        "xarelto": "rivaroxaban",
        "entresto": "sacubitril_valsartan",
        "keytruda": "pembrolizumab",
        "januvia": "sitagliptin",
        "trulicity": "dulaglutide",
        "mounjaro": "tirzepatide",
        "stelara": "ustekinumab",
        "skyrizi": "risankizumab",
        "lyrica": "pregabalin",
        "neurontin": "gabapentin",
        "cymbalta": "duloxetine",
        "zoloft": "sertraline",
        "lipitor": "atorvastatin",
        "crestor": "rosuvastatin",
        "coumadin": "warfarin",
        "norvasc": "amlodipine",
        "plavix": "clopidogrel",
        "prilosec": "omeprazole",
        "synthroid": "levothyroxine",
        "singulair": "montelukast",
        "revlimid": "lenalidomide",
    }
    cleaned = name.strip().lower().split()[0]
    return mapping.get(cleaned, cleaned)


@tool
def drug_interaction(drug: str, current_medications: str) -> str:
    """Check for drug-drug interactions between a new drug and current medications.

    Args:
        drug: The new drug being prescribed.
        current_medications: Comma-separated list of current medications.
    """
    new_drug = _normalize_drug_name(drug)
    med_list = [m.strip() for m in current_medications.split(",") if m.strip()]

    interactions = []
    max_severity = InteractionSeverity.NONE
    severity_order = {
        "none": 0, "minor": 1, "moderate": 2, "severe": 3,
    }

    for med in med_list:
        med_normalized = _normalize_drug_name(med)

        pair_key = (new_drug, med_normalized)
        reverse_key = (med_normalized, new_drug)

        interaction_data = INTERACTION_DB.get(pair_key) or INTERACTION_DB.get(reverse_key)

        if interaction_data:
            sev = InteractionSeverity(interaction_data["severity"])
            interactions.append(
                DrugInteraction(
                    drug_pair=f"{new_drug}-{med_normalized}",
                    severity=sev,
                    description=interaction_data["description"],
                )
            )
            if severity_order[sev.value] > severity_order[max_severity.value]:
                max_severity = sev
        else:
            interactions.append(
                DrugInteraction(
                    drug_pair=f"{new_drug}-{med_normalized}",
                    severity=InteractionSeverity.NONE,
                    description="No known interaction in database.",
                )
            )

    return InteractionResult(
        interactions=interactions,
        max_severity=max_severity,
    ).model_dump_json()
