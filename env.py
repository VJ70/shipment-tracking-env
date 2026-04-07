"""
ShipmentEnv — core environment logic.

Implements the full OpenEnv interface:
  reset(task_id)      → ResetResult
  step(session_id, action) → StepResult
  state(session_id)   → StateResult

Sessions are stored in-memory (dict). Each session is isolated.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

from models import (
    ShipmentAction,
    ShipmentObservation,
    StateResult,
    StepResult,
    ResetResult,
    TaskInfo,
)
from tools import ToolRegistry, TASK_TOOLS
from graders import get_grader

DATA_DIR = Path(__file__).parent / "data"

MAX_STEPS: Dict[str, int] = {
    "task1": 6,
    "task2": 9,
    "task3": 13,
}

TASKS: List[TaskInfo] = [
    TaskInfo(
        task_id="task1",
        name="Delay Notification",
        difficulty="easy",
        description=(
            "A shipment is delayed. The agent must check the shipment status "
            "and send a meaningful notification to the customer explaining the "
            "delay, then close the ticket."
        ),
        max_steps=MAX_STEPS["task1"],
        success_threshold=0.7,
        available_tools=TASK_TOOLS["task1"],
    ),
    TaskInfo(
        task_id="task2",
        name="Failed Delivery Rebook",
        difficulty="medium",
        description=(
            "A delivery attempt failed (no one home). The agent must verify "
            "shipment status, check carrier SLA compliance, rebook a new delivery "
            "date, notify the customer, and close the ticket."
        ),
        max_steps=MAX_STEPS["task2"],
        success_threshold=0.7,
        available_tools=TASK_TOOLS["task2"],
    ),
    TaskInfo(
        task_id="task3",
        name="Lost Package Full Resolution",
        difficulty="hard",
        description=(
            "A package is confirmed lost (7 days overdue). The agent must check "
            "status, verify carrier SLA breach, file an official carrier claim, "
            "issue a full refund for the correct order value, reship the order, "
            "notify the customer, and close the ticket with detailed notes."
        ),
        max_steps=MAX_STEPS["task3"],
        success_threshold=0.7,
        available_tools=TASK_TOOLS["task3"],
    ),
]

TASK_MAP = {t.task_id: t for t in TASKS}


class ShipmentEnv:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.tools = ToolRegistry()
        self._shipments = json.loads((DATA_DIR / "shipments.json").read_text())
        self._policies = json.loads((DATA_DIR / "policies.json").read_text())

    # ------------------------------------------------------------------ #
    # OpenEnv interface                                                     #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: str = "task1") -> ResetResult:
        """Start a new episode. Returns initial observation."""
        if task_id not in TASK_MAP:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(TASK_MAP.keys())}"
            )

        sid = str(uuid.uuid4())
        shipment = self._get_shipment(task_id)
        task = TASK_MAP[task_id]

        self.sessions[sid] = {
            "task_id": task_id,
            "shipment": shipment,
            "history": [],
            "turn": 0,
            "status": "open",
            "done": False,
            # tool outcome flags
            "customer_notified": False,
            "notification_message": "",
            "sla_checked": False,
            "rebooked": False,
            "rebook_date": None,
            "claim_filed": False,
            "claim_reason": "",
            "refund_issued": False,
            "refund_amount": 0.0,
            "reshipped": False,
            "resolution_notes": "",
        }

        tool_desc = self.tools.descriptions_for(task_id)
        tools_hint = "; ".join(
            f"{k}: {v}" for k, v in tool_desc.items()
        )

        obs = ShipmentObservation(
            shipment=shipment,
            tool_result=None,
            available_tools=task.available_tools,
            turn=0,
            message=(
                f"NEW TICKET — Shipment {shipment['shipment_id']} | "
                f"Status: {shipment['status'].upper()} | "
                f"Customer: {shipment['customer_name']} | "
                f"Order: {shipment['order_id']} | "
                f"Value: ${shipment['value']:.2f}. "
                f"Resolve this ticket using the available tools. "
                f"TOOLS: {tools_hint}"
            ),
        )
        return ResetResult(observation=obs, task_id=task_id, session_id=sid)

    def step(self, session_id: str, action: ShipmentAction) -> StepResult:
        """Execute one agent action. Returns observation, reward, done, info."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /reset first.")

        state = self.sessions[session_id]

        if state["done"]:
            raise RuntimeError("Episode is already finished. Call /reset to start a new one.")

        state["turn"] += 1
        task_id = state["task_id"]

        # Execute the tool
        tool_result = self.tools.call(action.tool_name, action.tool_args, state)

        # Log step
        state["history"].append({
            "turn": state["turn"],
            "tool": action.tool_name,
            "args": action.tool_args,
            "result": tool_result,
        })

        # Determine done
        done = (
            action.tool_name == "close_ticket"
            and state.get("status") == "closed"
        ) or state["turn"] >= MAX_STEPS[task_id]

        state["done"] = done

        # Compute reward
        grader = get_grader(task_id)
        reward = grader(state["history"], state, state["shipment"])
        reward = round(min(max(float(reward), 0.0), 1.0), 4)

        task = TASK_MAP[task_id]
        obs = ShipmentObservation(
            shipment=state["shipment"],
            tool_result=tool_result,
            available_tools=task.available_tools,
            turn=state["turn"],
            message=(
                "Ticket closed successfully."
                if done and state.get("status") == "closed"
                else f"Turn {state['turn']}/{MAX_STEPS[task_id]}. Continue resolving the shipment."
            ),
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "turn": state["turn"],
                "max_steps": MAX_STEPS[task_id],
                "task_id": task_id,
                "tool_called": action.tool_name,
            },
        )

    def get_state(self, session_id: str) -> StateResult:
        """Return current state snapshot for a session."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found.")
        s = self.sessions[session_id]
        return StateResult(
            session_id=session_id,
            task_id=s["task_id"],
            turn=s["turn"],
            status=s["status"],
            done=s["done"],
            shipment_id=s["shipment"]["shipment_id"],
            customer_notified=s["customer_notified"],
            rebooked=s["rebooked"],
            claim_filed=s["claim_filed"],
            refund_issued=s["refund_issued"],
            refund_amount=s["refund_amount"],
            reshipped=s["reshipped"],
            history_length=len(s["history"]),
        )

    def list_tasks(self) -> List[dict]:
        return [t.dict() for t in TASKS]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _get_shipment(self, task_id: str) -> dict:
        for s in self._shipments:
            if s["task_id"] == task_id:
                return dict(s)      # return a fresh copy per reset
        raise ValueError(f"No shipment found for task_id '{task_id}'")