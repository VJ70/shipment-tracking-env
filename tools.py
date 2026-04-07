"""
Tool library for the Shipment Tracking Agent Environment.
Each tool simulates a real logistics system API call.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).parent / "data"


def _load_policies() -> dict:
    return json.loads((DATA_DIR / "policies.json").read_text())


# Tools available per task (progressive unlock)
TASK_TOOLS: Dict[str, List[str]] = {
    "task1": [
        "get_shipment_status",
        "notify_customer",
        "close_ticket",
    ],
    "task2": [
        "get_shipment_status",
        "notify_customer",
        "check_carrier_sla",
        "rebook_delivery",
        "close_ticket",
    ],
    "task3": [
        "get_shipment_status",
        "notify_customer",
        "check_carrier_sla",
        "file_carrier_claim",
        "issue_refund",
        "reship_order",
        "close_ticket",
    ],
}

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "get_shipment_status": "Fetch current status, location, and details of the shipment. No args needed.",
    "notify_customer": "Send a notification to the customer. Args: customer_id (str), message (str).",
    "check_carrier_sla": "Check if the carrier has breached their SLA. No args needed.",
    "rebook_delivery": "Reschedule a failed delivery. Args: shipment_id (str), preferred_date (str YYYY-MM-DD).",
    "file_carrier_claim": "File an official claim with the carrier for a lost/damaged package. Args: shipment_id (str), carrier (str), reason (str).",
    "issue_refund": "Issue a refund to the customer. Args: order_id (str), amount (float), reason (str).",
    "reship_order": "Create a replacement shipment. Args: order_id (str), customer_id (str).",
    "close_ticket": "Close the support ticket. Args: resolution_notes (str).",
}


class ToolRegistry:

    def available_for(self, task_id: str) -> List[str]:
        return TASK_TOOLS.get(task_id, ["get_shipment_status", "close_ticket"])

    def descriptions_for(self, task_id: str) -> Dict[str, str]:
        return {t: TOOL_DESCRIPTIONS[t] for t in self.available_for(task_id)}

    def call(self, tool_name: str, args: Dict[str, Any], state: dict) -> str:
        allowed = self.available_for(state["task_id"])
        if tool_name not in allowed:
            return f"ERROR: tool '{tool_name}' is not available for {state['task_id']}. Available: {allowed}"

        fn = getattr(self, f"_tool_{tool_name}", None)
        if fn is None:
            return f"ERROR: tool '{tool_name}' not implemented"

        try:
            return fn(args, state)
        except Exception as exc:
            return f"ERROR: tool '{tool_name}' raised an exception: {exc}"

    # Individual tool implementations                                      

    def _tool_get_shipment_status(self, args: dict, state: dict) -> str:
        s = state["shipment"]
        policies = _load_policies()
        carrier_info = policies["carriers"].get(s["carrier"], {})
        return json.dumps({
            "shipment_id": s["shipment_id"],
            "order_id": s["order_id"],
            "status": s["status"],
            "carrier": s["carrier"],
            "carrier_sla_days": carrier_info.get("sla_days", 5),
            "current_location": s.get("current_location") or s.get("last_location", "unknown"),
            "expected_delivery": s["expected_delivery"],
            "delay_reason": s.get("delay_reason"),
            "failure_reason": s.get("failure_reason"),
            "days_overdue": s.get("days_overdue", 0),
            "customer_id": s["customer_id"],
            "customer_name": s["customer_name"],
            "order_value": s["value"],
            "items": s["items"],
            "insurance_eligible": s.get("insurance_eligible", False),
        }, indent=2)

    def _tool_notify_customer(self, args: dict, state: dict) -> str:
        missing = [k for k in ("customer_id", "message") if k not in args]
        if missing:
            return f"ERROR: missing required args: {missing}"

        s = state["shipment"]
        if args["customer_id"] != s["customer_id"]:
            return f"ERROR: customer_id '{args['customer_id']}' does not match shipment customer '{s['customer_id']}'"

        msg = str(args["message"]).strip()
        if len(msg) < 10:
            return "ERROR: message too short — provide a meaningful notification (min 10 chars)"

        state["customer_notified"] = True
        state["notification_message"] = msg
        return json.dumps({
            "status": "sent",
            "recipient": s["customer_email"],
            "customer_name": s["customer_name"],
            "message_preview": msg[:80],
        })

    def _tool_check_carrier_sla(self, args: dict, state: dict) -> str:
        s = state["shipment"]
        policies = _load_policies()
        carrier = s["carrier"]
        carrier_info = policies["carriers"].get(carrier, {"sla_days": 5})
        sla_days = carrier_info["sla_days"]
        overdue = s.get("days_overdue", 0)
        breached = overdue >= sla_days

        state["sla_checked"] = True
        return json.dumps({
            "carrier": carrier,
            "sla_days": sla_days,
            "days_overdue": overdue,
            "sla_breached": breached,
            "claim_eligible": overdue >= policies["claim_eligible_after_days"],
            "claim_url": carrier_info.get("claim_url", ""),
            "support_phone": carrier_info.get("support_phone", ""),
        })

    def _tool_rebook_delivery(self, args: dict, state: dict) -> str:
        missing = [k for k in ("shipment_id", "preferred_date") if k not in args]
        if missing:
            return f"ERROR: missing required args: {missing}"

        s = state["shipment"]
        if args["shipment_id"] != s["shipment_id"]:
            return f"ERROR: shipment_id mismatch — expected '{s['shipment_id']}'"

        if s["status"] not in ("failed_delivery", "delayed"):
            return f"ERROR: cannot rebook — shipment status is '{s['status']}'"

        attempts = s.get("rebook_attempts", 0)
        if attempts >= 3:
            return "ERROR: maximum rebook attempts (3) reached — escalate to lost package process"

        pdate = str(args["preferred_date"])
        state["rebooked"] = True
        state["rebook_date"] = pdate
        state["shipment"]["rebook_attempts"] = attempts + 1

        return json.dumps({
            "status": "rebooked",
            "shipment_id": s["shipment_id"],
            "new_delivery_date": pdate,
            "attempts_used": attempts + 1,
            "attempts_remaining": 3 - (attempts + 1),
        })

    def _tool_file_carrier_claim(self, args: dict, state: dict) -> str:
        missing = [k for k in ("shipment_id", "carrier", "reason") if k not in args]
        if missing:
            return f"ERROR: missing required args: {missing}"

        s = state["shipment"]
        policies = _load_policies()

        if args["shipment_id"] != s["shipment_id"]:
            return f"ERROR: shipment_id mismatch — expected '{s['shipment_id']}'"

        days_overdue = s.get("days_overdue", 0)
        threshold = policies["claim_eligible_after_days"]
        if days_overdue < threshold:
            return (
                f"ERROR: claim not yet eligible — package must be {threshold}+ days overdue "
                f"(currently {days_overdue} days). Wait before filing."
            )

        state["claim_filed"] = True
        state["claim_reason"] = str(args["reason"])
        claim_id = f"CLM-{s['shipment_id']}-{s['carrier'][:2].upper()}"

        return json.dumps({
            "status": "claim_filed",
            "claim_id": claim_id,
            "shipment_id": s["shipment_id"],
            "carrier": args["carrier"],
            "reason": args["reason"],
            "expected_resolution_days": "5-7 business days",
            "next_step": "Issue refund or reship while claim is processed",
        })

    def _tool_issue_refund(self, args: dict, state: dict) -> str:
        missing = [k for k in ("order_id", "amount", "reason") if k not in args]
        if missing:
            return f"ERROR: missing required args: {missing}"

        s = state["shipment"]
        if args["order_id"] != s["order_id"]:
            return f"ERROR: order_id mismatch — expected '{s['order_id']}'"

        try:
            amount = float(args["amount"])
        except (TypeError, ValueError):
            return "ERROR: 'amount' must be a numeric value"

        if amount <= 0:
            return "ERROR: refund amount must be greater than 0"

        if amount > s["value"]:
            return (
                f"ERROR: refund amount ${amount:.2f} exceeds order value "
                f"${s['value']:.2f}. Maximum refund is ${s['value']:.2f}."
            )

        state["refund_issued"] = True
        state["refund_amount"] = amount

        return json.dumps({
            "status": "refund_issued",
            "order_id": args["order_id"],
            "amount_refunded": round(amount, 2),
            "order_value": s["value"],
            "reason": args["reason"],
            "processing_days": "3-5 business days",
        })

    def _tool_reship_order(self, args: dict, state: dict) -> str:
        missing = [k for k in ("order_id", "customer_id") if k not in args]
        if missing:
            return f"ERROR: missing required args: {missing}"

        s = state["shipment"]
        if args["order_id"] != s["order_id"]:
            return f"ERROR: order_id mismatch — expected '{s['order_id']}'"
        if args["customer_id"] != s["customer_id"]:
            return f"ERROR: customer_id mismatch — expected '{s['customer_id']}'"

        # Enforce logical ordering: must claim or refund first
        if not state.get("claim_filed") and not state.get("refund_issued"):
            return (
                "ERROR: must file a carrier claim (file_carrier_claim) or issue a refund "
                "(issue_refund) before reshipping. This protects against duplicate shipments."
            )

        state["reshipped"] = True
        suffix = s["shipment_id"].replace("SHP", "")
        new_id = f"SHP{suffix}-RESHIP"

        return json.dumps({
            "status": "reshipped",
            "original_shipment_id": s["shipment_id"],
            "new_shipment_id": new_id,
            "order_id": s["order_id"],
            "items": s["items"],
            "estimated_delivery": "3-5 business days",
        })

    def _tool_close_ticket(self, args: dict, state: dict) -> str:
        if "resolution_notes" not in args:
            return "ERROR: missing required arg 'resolution_notes'"

        notes = str(args["resolution_notes"]).strip()
        if len(notes) < 10:
            return "ERROR: resolution_notes too short — describe what was done (min 10 chars)"

        state["status"] = "closed"
        state["resolution_notes"] = notes

        return json.dumps({
            "status": "closed",
            "resolution_notes": notes,
            "message": "Ticket closed successfully.",
        })