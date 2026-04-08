"""
Graders for the Shipment Tracking Agent Environment.
All graders return a float STRICTLY between 0.0 and 1.0 (exclusive).
i.e. output is always in the open interval (0.0, 1.0).

Scoring philosophy:
  - Partial credit for each correct action taken
  - Full credit only when all steps are completed properly
  - Penalise wrong refund amounts but still award partial credit
  - Ticket must be closed to earn the final component
"""
from __future__ import annotations
from typing import Any, Dict, List

EPSILON = 1e-6   # ensures score is never exactly 0.0 or 1.0


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.0, 1.0)."""
    return round(min(max(score, EPSILON), 1.0 - EPSILON), 6)


def grade_task1(history: List[dict], state: Dict[str, Any], shipment: dict) -> float:
    """
    Task 1 (Easy) — Delay Notification.

    Components:
      1. Check the shipment status              -> 0.30
      2. Notify the customer with a real message -> 0.40
      3. Close the ticket                        -> 0.29
    Max raw = 0.99 -> _clamp keeps it strictly below 1.0
    Min raw = 0.0  -> _clamp raises it to 1e-6 (strictly above 0.0)
    """
    score = 0.0

    if any(s["tool"] == "get_shipment_status" for s in history):
        score += 0.30

    if state.get("customer_notified", False):
        msg = state.get("notification_message", "")
        if len(msg) >= 20:
            score += 0.40
        else:
            score += 0.15       # partial: notified but message too brief

    if state.get("status") == "closed" and state.get("resolution_notes"):
        score += 0.29           # 0.29 not 0.30 so max raw = 0.99

    return _clamp(score)


def grade_task2(history: List[dict], state: Dict[str, Any], shipment: dict) -> float:
    """
    Task 2 (Medium) — Failed Delivery Rebook.

    Components:
      1. Check shipment status   -> 0.20
      2. Check carrier SLA       -> 0.20
      3. Rebook delivery         -> 0.25
      4. Notify customer         -> 0.14
      5. Close the ticket        -> 0.20
    Max raw = 0.99 -> _clamp keeps it strictly below 1.0
    """
    score = 0.0

    if any(s["tool"] == "get_shipment_status" for s in history):
        score += 0.20

    if any(s["tool"] == "check_carrier_sla" for s in history) or state.get("sla_checked"):
        score += 0.20

    if state.get("rebooked", False):
        score += 0.25

    if state.get("customer_notified", False):
        msg = state.get("notification_message", "")
        score += 0.14 if len(msg) >= 20 else 0.07   # 0.14 keeps max = 0.99

    if state.get("status") == "closed" and state.get("resolution_notes"):
        score += 0.20

    return _clamp(score)


def grade_task3(history: List[dict], state: Dict[str, Any], shipment: dict) -> float:
    """
    Task 3 (Hard) — Lost Package Resolution.

    Components:
      1. Check shipment status   -> 0.10
      2. Check carrier SLA       -> 0.10
      3. File a carrier claim    -> 0.20
      4. Issue correct refund    -> 0.24
      5. Reship the order        -> 0.20
      6. Notify the customer     -> 0.05
      7. Close the ticket        -> 0.10
    Max raw = 0.99 -> _clamp keeps it strictly below 1.0
    """
    score = 0.0

    if any(s["tool"] == "get_shipment_status" for s in history):
        score += 0.10

    if any(s["tool"] == "check_carrier_sla" for s in history) or state.get("sla_checked"):
        score += 0.10

    if state.get("claim_filed", False):
        score += 0.20

    if state.get("refund_issued", False):
        expected = float(shipment.get("value", 0))
        actual   = float(state.get("refund_amount", 0))
        if expected > 0:
            error_pct = abs(actual - expected) / expected
            if error_pct <= 0.05:
                score += 0.24       # 0.24 so max raw stays 0.99
            elif error_pct <= 0.20:
                score += 0.15
            else:
                score += 0.08
        else:
            score += 0.24

    if state.get("reshipped", False):
        score += 0.20

    if state.get("customer_notified", False):
        score += 0.05

    if state.get("status") == "closed" and len(state.get("resolution_notes", "")) >= 10:
        score += 0.10

    return _clamp(score)


def get_grader(task_id: str):
    """Return the grader function for a given task_id."""
    graders = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(graders.keys())}")
    return graders[task_id]


def get_grader(task_id: str):
    """Return the grader function for a given task_id."""
    graders = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(graders.keys())}")
    return graders[task_id]