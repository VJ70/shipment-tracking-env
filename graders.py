"""
Graders for the Shipment Tracking Agent Environment.
All graders are deterministic and return a float in (0.0, 1.0) exclusive.

Scoring philosophy:
  - Partial credit for each correct action taken
  - Full credit only when all steps are completed properly
  - Penalise wrong refund amounts but still award partial credit
  - Ticket must be closed to earn the final component
"""
from __future__ import annotations
from typing import Any, Dict, List


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.0, 1.0)."""
    EPSILON = 1e-6
    return round(min(max(score, EPSILON), 1.0 - EPSILON), 6)


def grade_task1(history: List[dict], state: Dict[str, Any], shipment: dict) -> float:
    """
    Task 1 (Easy) — Delay Notification.

    The agent must:
      1. Check the shipment status              → 0.30
      2. Notify the customer with a real message → 0.40
      3. Close the ticket                        → 0.30
    Max = 1.0 (clamped to <1.0)
    """
    score = 0.0

    # Component 1: checked shipment status
    if any(s["tool"] == "get_shipment_status" for s in history):
        score += 0.30

    # Component 2: notified customer with a substantive message
    if state.get("customer_notified", False):
        msg = state.get("notification_message", "")
        if len(msg) >= 20:          # meaningful message, not just "ok"
            score += 0.40
        else:
            score += 0.15           # partial: notified but message too brief

    # Component 3: closed the ticket
    if state.get("status") == "closed" and state.get("resolution_notes"):
        score += 0.30

    return _clamp(score)


def grade_task2(history: List[dict], state: Dict[str, Any], shipment: dict) -> float:
    """
    Task 2 (Medium) — Failed Delivery Rebook.

    The agent must:
      1. Check shipment status                  → 0.20
      2. Check carrier SLA                      → 0.20
      3. Rebook the delivery with a date         → 0.25
      4. Notify the customer about the rebook    → 0.15
      5. Close the ticket                        → 0.20
    Max = 1.0 (clamped to <1.0)
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
        score += 0.15 if len(msg) >= 20 else 0.07

    if state.get("status") == "closed" and state.get("resolution_notes"):
        score += 0.20

    return _clamp(score)


def grade_task3(history: List[dict], state: Dict[str, Any], shipment: dict) -> float:
    """
    Task 3 (Hard) — Lost Package Resolution.

    The agent must:
      1. Check shipment status                  → 0.10
      2. Check carrier SLA                      → 0.10
      3. File a carrier claim                   → 0.20
      4. Issue correct refund                   → 0.25  (partial if wrong amount)
      5. Reship the order                       → 0.20
      6. Notify the customer                    → 0.05
      7. Close the ticket with notes            → 0.10
    Max = 1.0 (clamped to <1.0)
    """
    score = 0.0

    if any(s["tool"] == "get_shipment_status" for s in history):
        score += 0.10

    if any(s["tool"] == "check_carrier_sla" for s in history) or state.get("sla_checked"):
        score += 0.10

    if state.get("claim_filed", False):
        score += 0.20

    # Refund scoring: full credit if within 5%, partial if issued but wrong amount
    if state.get("refund_issued", False):
        expected = float(shipment.get("value", 0))
        actual = float(state.get("refund_amount", 0))
        if expected > 0:
            error_pct = abs(actual - expected) / expected
            if error_pct <= 0.05:
                score += 0.25       # correct amount
            elif error_pct <= 0.20:
                score += 0.15       # within 20% tolerance
            else:
                score += 0.08       # refund issued but significantly wrong
        else:
            score += 0.25           # no expected value to compare

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