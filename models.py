"""
Typed Pydantic models for the Shipment Tracking Agent Environment.
Complies with OpenEnv spec: Action, Observation, StepResult, ResetResult.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ShipmentAction(BaseModel):
    """Action sent by the agent each step."""
    tool_name: str = Field(..., description="Name of the tool to call")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class ShipmentObservation(BaseModel):
    """Observation returned to the agent after each step."""
    shipment: Dict[str, Any] = Field(..., description="Current shipment details")
    tool_result: Optional[str] = Field(None, description="Result from last tool call")
    available_tools: List[str] = Field(..., description="Tools available in this task")
    turn: int = Field(..., description="Current turn number (0 = reset)")
    message: str = Field(..., description="Human-readable status message")


class StepResult(BaseModel):
    """Returned by step() — full OpenEnv spec."""
    observation: ShipmentObservation
    reward: float = Field(..., ge=0.0, le=1.0, description="Reward signal [0.0, 1.0]")
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra diagnostic info")


class ResetResult(BaseModel):
    """Returned by reset() — full OpenEnv spec."""
    observation: ShipmentObservation
    task_id: str
    session_id: str


class StateResult(BaseModel):
    """Returned by state() — full OpenEnv spec."""
    session_id: str
    task_id: str
    turn: int
    status: str
    done: bool
    shipment_id: str
    customer_notified: bool
    rebooked: bool
    claim_filed: bool
    refund_issued: bool
    refund_amount: float
    reshipped: bool
    history_length: int


class TaskInfo(BaseModel):
    """Metadata for a single task."""
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    success_threshold: float
    available_tools: List[str]