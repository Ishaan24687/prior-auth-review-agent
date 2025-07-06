"""
Drug cost estimation tool.

AWP (Average Wholesale Price) based pricing with typical PBM discount and
copay/coinsurance schedules. Also suggests cheaper therapeutic alternatives
when available.

# TODO: integrate real-time NDC pricing via Medi-Span or First Databank
# TODO: add accumulator/deductible tracking per member
"""

import json
from typing import Any

from langchain.tools import tool

DRUG_PRICING: dict[str, dict[str, Any]] = {
    "metformin": {
        "awp_per_unit": 0.15,
        "units_per_fill": 180,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "atorvastatin": {
        "awp_per_unit": 0.30,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "lisinopril": {
        "awp_per_unit": 0.12,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "gabapentin": {
        "awp_per_unit": 0.18,
        "units_per_fill": 270,
        "copay_tier1": 10.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "sertraline": {
        "awp_per_unit": 0.22,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "omeprazole": {
        "awp_per_unit": 0.25,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "losartan": {
        "awp_per_unit": 0.20,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "amlodipine": {
        "awp_per_unit": 0.10,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
    "rosuvastatin": {
        "awp_per_unit": 0.80,
        "units_per_fill": 30,
        "copay_tier1": 15.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [
            {"drug": "atorvastatin", "monthly_cost": 9.00, "tier": 1},
        ],
    },
    "pregabalin": {
        "awp_per_unit": 2.50,
        "units_per_fill": 60,
        "copay_tier1": 25.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [
            {"drug": "gabapentin", "monthly_cost": 48.60, "tier": 1},
        ],
    },
    "ozempic": {
        "awp_per_unit": 223.00,
        "units_per_fill": 4,
        "copay_tier1": 75.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "trulicity", "monthly_cost": 780.00, "tier": 4},
            {"drug": "jardiance", "monthly_cost": 540.00, "tier": 3},
        ],
    },
    "jardiance": {
        "awp_per_unit": 18.00,
        "units_per_fill": 30,
        "copay_tier1": 50.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "farxiga", "monthly_cost": 510.00, "tier": 3},
        ],
    },
    "farxiga": {
        "awp_per_unit": 17.00,
        "units_per_fill": 30,
        "copay_tier1": 50.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "jardiance", "monthly_cost": 540.00, "tier": 3},
        ],
    },
    "eliquis": {
        "awp_per_unit": 8.50,
        "units_per_fill": 60,
        "copay_tier1": 40.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "xarelto", "monthly_cost": 480.00, "tier": 3},
            {"drug": "warfarin", "monthly_cost": 4.50, "tier": 1},
        ],
    },
    "xarelto": {
        "awp_per_unit": 16.00,
        "units_per_fill": 30,
        "copay_tier1": 40.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "eliquis", "monthly_cost": 510.00, "tier": 3},
            {"drug": "warfarin", "monthly_cost": 4.50, "tier": 1},
        ],
    },
    "entresto": {
        "awp_per_unit": 10.50,
        "units_per_fill": 60,
        "copay_tier1": 60.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "lisinopril", "monthly_cost": 3.60, "tier": 1},
            {"drug": "losartan", "monthly_cost": 6.00, "tier": 1},
        ],
    },
    "humira": {
        "awp_per_unit": 3400.00,
        "units_per_fill": 2,
        "copay_tier1": 150.00,
        "coinsurance_pct": 0.20,
        "generic_available": False,
        "alternatives": [
            {"drug": "enbrel", "monthly_cost": 6200.00, "tier": 5},
        ],
    },
    "enbrel": {
        "awp_per_unit": 1550.00,
        "units_per_fill": 4,
        "copay_tier1": 150.00,
        "coinsurance_pct": 0.20,
        "generic_available": False,
        "alternatives": [
            {"drug": "humira", "monthly_cost": 6800.00, "tier": 5},
        ],
    },
    "keytruda": {
        "awp_per_unit": 10166.00,
        "units_per_fill": 1,
        "copay_tier1": 0.00,
        "coinsurance_pct": 0.20,
        "generic_available": False,
        "alternatives": [],
    },
    "stelara": {
        "awp_per_unit": 13000.00,
        "units_per_fill": 1,
        "copay_tier1": 0.00,
        "coinsurance_pct": 0.20,
        "generic_available": False,
        "alternatives": [
            {"drug": "skyrizi", "monthly_cost": 5500.00, "tier": 5},
        ],
    },
    "skyrizi": {
        "awp_per_unit": 16500.00,
        "units_per_fill": 1,
        "copay_tier1": 0.00,
        "coinsurance_pct": 0.20,
        "generic_available": False,
        "alternatives": [
            {"drug": "stelara", "monthly_cost": 4333.00, "tier": 5},
        ],
    },
    "revlimid": {
        "awp_per_unit": 850.00,
        "units_per_fill": 21,
        "copay_tier1": 0.00,
        "coinsurance_pct": 0.25,
        "generic_available": False,
        "alternatives": [],
    },
    "wegovy": {
        "awp_per_unit": 340.00,
        "units_per_fill": 4,
        "copay_tier1": 0.00,
        "coinsurance_pct": 1.0,
        "generic_available": False,
        "alternatives": [],
    },
    "mounjaro": {
        "awp_per_unit": 262.00,
        "units_per_fill": 4,
        "copay_tier1": 75.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "ozempic", "monthly_cost": 892.00, "tier": 3},
        ],
    },
    "trulicity": {
        "awp_per_unit": 195.00,
        "units_per_fill": 4,
        "copay_tier1": 75.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "ozempic", "monthly_cost": 892.00, "tier": 3},
        ],
    },
    "januvia": {
        "awp_per_unit": 15.50,
        "units_per_fill": 30,
        "copay_tier1": 50.00,
        "coinsurance_pct": 0.0,
        "generic_available": False,
        "alternatives": [
            {"drug": "jardiance", "monthly_cost": 540.00, "tier": 3},
        ],
    },
    "duloxetine": {
        "awp_per_unit": 1.80,
        "units_per_fill": 30,
        "copay_tier1": 15.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [
            {"drug": "sertraline", "monthly_cost": 6.60, "tier": 1},
        ],
    },
    "warfarin": {
        "awp_per_unit": 0.15,
        "units_per_fill": 30,
        "copay_tier1": 5.00,
        "coinsurance_pct": 0.0,
        "generic_available": True,
        "alternatives": [],
    },
}

BRAND_TO_GENERIC: dict[str, str] = {
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "norvasc": "amlodipine",
    "zoloft": "sertraline",
    "cymbalta": "duloxetine",
    "prilosec": "omeprazole",
    "neurontin": "gabapentin",
    "lyrica": "pregabalin",
    "plavix": "clopidogrel",
    "synthroid": "levothyroxine",
    "coumadin": "warfarin",
    "singulair": "montelukast",
}


@tool
def cost_estimator(drug: str, quantity: int = 30, days_supply: int = 30) -> str:
    """Estimate the cost of a drug for the plan and member.

    Args:
        drug: Drug name (brand or generic).
        quantity: Number of units requested.
        days_supply: Days supply requested.
    """
    normalized = drug.strip().lower()
    normalized = BRAND_TO_GENERIC.get(normalized, normalized)

    if normalized not in DRUG_PRICING:
        return json.dumps({
            "drug_name": drug,
            "error": f"Drug '{drug}' not found in pricing database.",
        })

    pricing = DRUG_PRICING[normalized]
    awp_total = pricing["awp_per_unit"] * quantity
    fills_per_year = 365 / max(days_supply, 1)

    if pricing["coinsurance_pct"] > 0:
        member_cost = awp_total * pricing["coinsurance_pct"]
    else:
        member_cost = pricing["copay_tier1"]

    plan_cost = awp_total - member_cost

    return json.dumps({
        "drug_name": drug,
        "plan_cost_30day": round(plan_cost, 2),
        "member_copay": round(member_cost, 2),
        "member_coinsurance_pct": pricing["coinsurance_pct"],
        "total_annual_cost": round(awp_total * fills_per_year, 2),
        "cheaper_alternatives": pricing["alternatives"],
    })
