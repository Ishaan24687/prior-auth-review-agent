"""
Member eligibility verification tool.

Hardcoded member database with 20 members across different plan types,
eligibility statuses, and pharmacy benefit configurations.

# TODO: integrate real-time eligibility API via X12 270/271 transactions
# TODO: add dependent eligibility checks (e.g., member under parent's plan)
"""

from langchain.tools import tool

from src.graph.state import EligibilityResult

MEMBER_DB: dict[str, dict] = {
    "M001": {
        "member_name": "James Wilson",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M002": {
        "member_name": "Sarah Chen",
        "plan_active": True,
        "plan_type": "HMO",
        "effective_date": "2024-03-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M003": {
        "member_name": "Robert Martinez",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2023-06-01",
        "termination_date": "2025-05-31",
        "pharmacy_benefit_active": True,
    },
    "M004": {
        "member_name": "Emily Johnson",
        "plan_active": False,
        "plan_type": "EPO",
        "effective_date": "2023-01-01",
        "termination_date": "2023-12-31",
        "pharmacy_benefit_active": False,
    },
    "M005": {
        "member_name": "Michael Brown",
        "plan_active": True,
        "plan_type": "HMO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M006": {
        "member_name": "Jessica Lee",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": False,
    },
    "M007": {
        "member_name": "David Kim",
        "plan_active": True,
        "plan_type": "HMO",
        "effective_date": "2024-02-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M008": {
        "member_name": "Amanda Rivera",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-15",
        "termination_date": "2025-01-14",
        "pharmacy_benefit_active": True,
    },
    "M009": {
        "member_name": "Thomas Wright",
        "plan_active": True,
        "plan_type": "EPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M010": {
        "member_name": "Patricia Hernandez",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2023-07-01",
        "termination_date": "2024-06-30",
        "pharmacy_benefit_active": True,
    },
    "M011": {
        "member_name": "Christopher Davis",
        "plan_active": False,
        "plan_type": "HMO",
        "effective_date": "2022-01-01",
        "termination_date": "2023-06-30",
        "pharmacy_benefit_active": False,
    },
    "M012": {
        "member_name": "Jennifer Taylor",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M013": {
        "member_name": "Daniel Anderson",
        "plan_active": True,
        "plan_type": "HMO",
        "effective_date": "2024-04-01",
        "termination_date": "2025-03-31",
        "pharmacy_benefit_active": True,
    },
    "M014": {
        "member_name": "Laura Thomas",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M015": {
        "member_name": "Kevin Jackson",
        "plan_active": True,
        "plan_type": "EPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M016": {
        "member_name": "Stephanie White",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-05-01",
        "termination_date": "2025-04-30",
        "pharmacy_benefit_active": True,
    },
    "M017": {
        "member_name": "Brian Harris",
        "plan_active": True,
        "plan_type": "HMO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M018": {
        "member_name": "Nicole Clark",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-01",
        "termination_date": "2024-12-31",
        "pharmacy_benefit_active": True,
    },
    "M019": {
        "member_name": "Mark Lewis",
        "plan_active": False,
        "plan_type": "HMO",
        "effective_date": "2023-01-01",
        "termination_date": "2024-03-15",
        "pharmacy_benefit_active": False,
    },
    "M020": {
        "member_name": "Rachel Robinson",
        "plan_active": True,
        "plan_type": "PPO",
        "effective_date": "2024-01-01",
        "termination_date": "2025-12-31",
        "pharmacy_benefit_active": True,
    },
}


@tool
def eligibility_check(member_id: str) -> str:
    """Check a member's plan eligibility and pharmacy benefit status.

    Args:
        member_id: The plan member identifier (e.g., 'M001').
    """
    normalized = member_id.strip().upper()

    if normalized in MEMBER_DB:
        entry = MEMBER_DB[normalized]
        result = EligibilityResult(**entry)
        return result.model_dump_json()

    return EligibilityResult(
        plan_active=False,
        error=f"Member '{member_id}' not found in system. Verify member ID and resubmit.",
    ).model_dump_json()
