"""
Evaluation harness — compare agent decisions against expected outcomes.

Runs the deterministic graph workflow (no LLM calls) over the 25-case
test suite and calculates per-class precision, recall, and overall accuracy.

Usage:
    python -m src.evaluation.accuracy
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.graph.state import PARequest, UrgencyLevel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def load_test_cases(path: Path = TEST_CASES_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def run_single_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run the deterministic workflow on a single test case and return
    the predicted decision alongside the expected one."""
    from src.agents.reviewer import run_deterministic_review

    request = PARequest(
        member_id=case["member_id"],
        drug_name=case["drug"],
        diagnosis_codes=case["diagnosis_codes"],
        current_medications=case.get("current_medications", []),
        provider_npi=case.get("provider_npi", ""),
        quantity=case.get("quantity", 30),
        days_supply=case.get("days_supply", 30),
        urgency=UrgencyLevel.STANDARD,
    )

    result = run_deterministic_review(request)
    decision_data = result.get("decision", {})

    if isinstance(decision_data, dict):
        predicted = decision_data.get("decision", "unknown")
    else:
        predicted = str(decision_data)

    return {
        "id": case["id"],
        "description": case["description"],
        "drug": case["drug"],
        "expected": case["expected_decision"],
        "predicted": predicted,
        "match": predicted == case["expected_decision"],
        "summary": result.get("summary", "")[:200],
    }


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate accuracy, precision, recall, and F1 per decision class."""
    classes = ["approve", "deny", "pend"]
    total = len(results)
    correct = sum(1 for r in results if r["match"])

    # per-class TP, FP, FN
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for r in results:
        expected = r["expected"]
        predicted = r["predicted"]
        if predicted == expected:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

    metrics: dict[str, Any] = {
        "total_cases": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_class": {},
    }

    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics["per_class"][cls] = {
            "tp": tp[cls],
            "fp": fp[cls],
            "fn": fn[cls],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return metrics


def print_results(results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    """Pretty-print evaluation results to stdout."""
    print("\n" + "=" * 70)
    print("PRIOR AUTH REVIEW AGENT — EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'ID':<10} {'Drug':<15} {'Expected':<10} {'Predicted':<10} {'Match'}")
    print("-" * 60)
    for r in results:
        match_icon = "✓" if r["match"] else "✗"
        print(f"{r['id']:<10} {r['drug']:<15} {r['expected']:<10} {r['predicted']:<10} {match_icon}")

    print(f"\n{'─' * 40}")
    print(f"Overall Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total_cases']})")
    print(f"{'─' * 40}")

    print(f"\n{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<5} {'FP':<5} {'FN':<5}")
    print("-" * 60)
    for cls, m in metrics["per_class"].items():
        print(
            f"{cls:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
            f"{m['f1']:<12.4f} {m['tp']:<5} {m['fp']:<5} {m['fn']:<5}"
        )

    # Mismatches
    mismatches = [r for r in results if not r["match"]]
    if mismatches:
        print(f"\n{'─' * 40}")
        print(f"MISMATCHES ({len(mismatches)}):")
        for r in mismatches:
            print(f"  {r['id']} ({r['drug']}): expected={r['expected']}, got={r['predicted']}")
            print(f"    {r['description']}")
    else:
        print("\nNo mismatches — all cases predicted correctly.")

    print()


def main():
    logger.info("Loading test cases from %s", TEST_CASES_PATH)
    cases = load_test_cases()
    logger.info("Running %d test cases...", len(cases))

    results = []
    for i, case in enumerate(cases, 1):
        logger.info("[%d/%d] %s — %s", i, len(cases), case["id"], case["drug"])
        try:
            result = run_single_case(case)
            results.append(result)
        except Exception as e:
            logger.error("Case %s failed: %s", case["id"], e)
            results.append({
                "id": case["id"],
                "description": case["description"],
                "drug": case["drug"],
                "expected": case["expected_decision"],
                "predicted": "error",
                "match": False,
                "summary": str(e)[:200],
            })

    metrics = calculate_metrics(results)
    print_results(results, metrics)

    output_path = Path("eval_results")
    output_path.mkdir(exist_ok=True)
    with open(output_path / "results.json", "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)
    logger.info("Results saved to %s", output_path / "results.json")


if __name__ == "__main__":
    main()
