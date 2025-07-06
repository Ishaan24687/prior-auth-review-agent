"""
ICD-10-CM code validator.

Contains 50+ of the most commonly encountered ICD-10 codes in PA review,
covering endocrine, cardiovascular, musculoskeletal, dermatologic, oncologic,
and pain conditions.

# TODO: integrate full ICD-10-CM code set via CMS API or local database
# TODO: add ICD-10-PCS procedure code validation
"""

from langchain.tools import tool

ICD10_CODES: dict[str, str] = {
    # Endocrine / Metabolic
    "E11": "Type 2 diabetes mellitus",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
    "E11.21": "Type 2 diabetes mellitus with diabetic nephropathy",
    "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
    "E11.40": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
    "E11.42": "Type 2 diabetes mellitus with diabetic polyneuropathy",
    "E11.51": "Type 2 diabetes mellitus with diabetic peripheral angiopathy without gangrene",
    "E10.9": "Type 1 diabetes mellitus without complications",
    "E10.65": "Type 1 diabetes mellitus with hyperglycemia",
    "E66.01": "Morbid (severe) obesity due to excess calories",
    "E66.09": "Other obesity due to excess calories",
    "E78.00": "Pure hypercholesterolemia, unspecified",
    "E78.5": "Dyslipidemia, unspecified",
    "E03.9": "Hypothyroidism, unspecified",
    # Cardiovascular
    "I10": "Essential (primary) hypertension",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "I48.0": "Paroxysmal atrial fibrillation",
    "I48.1": "Persistent atrial fibrillation",
    "I48.91": "Unspecified atrial fibrillation",
    "I50.20": "Unspecified systolic (congestive) heart failure",
    "I50.22": "Chronic systolic (congestive) heart failure",
    "I50.9": "Heart failure, unspecified",
    "I26.99": "Other pulmonary embolism without acute cor pulmonale",
    "I82.40": "Acute embolism and thrombosis of unspecified deep veins of lower extremity",
    "I82.90": "Acute embolism and thrombosis of unspecified vein",
    "I63.9": "Cerebral infarction, unspecified",
    "I21.9": "Acute myocardial infarction, unspecified",
    # Musculoskeletal / Rheumatologic
    "M05.79": "Rheumatoid arthritis with rheumatoid factor of unspecified site",
    "M06.9": "Rheumatoid arthritis, unspecified",
    "M06.09": "Rheumatoid arthritis without rheumatoid factor, unspecified site",
    "M54.5": "Low back pain",
    "M79.3": "Panniculitis, unspecified",
    # Dermatologic
    "L40.0": "Psoriasis vulgaris",
    "L40.1": "Generalized pustular psoriasis",
    "L40.4": "Guttate psoriasis",
    "L40.50": "Arthropathic psoriasis, unspecified",
    # GI
    "K21.0": "Gastro-esophageal reflux disease with esophagitis",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "K50.90": "Crohn's disease, unspecified, without complications",
    "K51.90": "Ulcerative colitis, unspecified, without complications",
    # Oncology
    "C43.9": "Malignant melanoma of skin, unspecified",
    "C43.4": "Malignant melanoma of scalp and neck",
    "C43.5": "Malignant melanoma of trunk",
    "C90.00": "Multiple myeloma not having achieved remission",
    "C34.90": "Malignant neoplasm of unspecified part of unspecified bronchus or lung",
    # Pain / Neuro
    "G89.29": "Other chronic pain",
    "G62.9": "Polyneuropathy, unspecified",
    "G63": "Polyneuropathy in diseases classified elsewhere",
    "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
    # Renal
    "N18.3": "Chronic kidney disease, stage 3 (moderate)",
    "N18.4": "Chronic kidney disease, stage 4 (severe)",
    "N18.9": "Chronic kidney disease, unspecified",
    # Other
    "Z79.01": "Long term (current) use of anticoagulants",
    "Z79.02": "Long term (current) use of antithrombotics/antiplatelets",
    "Z79.4": "Long term (current) use of insulin",
    "Z79.84": "Long term (current) use of oral hypoglycemic drugs",
    "Z87.39": "Other personal history of other diseases of the musculoskeletal system and connective tissue",
}


@tool
def icd10_validator(codes: str) -> str:
    """Validate ICD-10-CM diagnosis codes and return descriptions.

    Args:
        codes: Comma-separated list of ICD-10-CM codes to validate.
    """
    code_list = [c.strip().upper() for c in codes.split(",") if c.strip()]
    results = []
    all_valid = True

    for code in code_list:
        if code in ICD10_CODES:
            results.append({
                "code": code,
                "valid": True,
                "description": ICD10_CODES[code],
            })
        else:
            prefix_match = None
            for known_code, desc in ICD10_CODES.items():
                if code.startswith(known_code) or known_code.startswith(code):
                    prefix_match = {"code": code, "valid": True, "description": desc, "note": f"Matched via prefix to {known_code}"}
                    break
            if prefix_match:
                results.append(prefix_match)
            else:
                results.append({
                    "code": code,
                    "valid": False,
                    "description": "Code not found in database",
                })
                all_valid = False

    import json
    return json.dumps({
        "all_valid": all_valid,
        "codes": results,
    })
