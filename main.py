"""
FastAPI application — Shipment Tracking Agent Environment.

Endpoints (OpenEnv spec):
  GET  /health          → liveness probe
  GET  /tasks           → list all tasks
  POST /reset           → start new episode
  POST /step            → execute one agent action
  GET  /state           → current session state
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from env import ShipmentEnv
from models import ShipmentAction, ResetResult, StepResult, StateResult

app = FastAPI(
    title="Shipment Tracking Agent Environment",
    description=(
        "An OpenEnv-compliant environment where an AI agent resolves real-world "
        "shipment exceptions: delays, failed deliveries, and lost packages."
    ),
    version="1.0.0",
)

env = ShipmentEnv()


@app.get("/health")
def health():
    """Liveness probe — must return 200 for deployment validation."""
    return {"status": "ok", "env": "shipment-tracking-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return env.list_tasks()


@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = Query(default="task1", description="task1 | task2 | task3")):
    """
    Start a new episode.
    Returns initial observation and a session_id for subsequent step() calls.
    """
    try:
        return env.reset(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult)
def step(
    session_id: str = Query(..., description="Session ID from /reset"),
    action: ShipmentAction = ...,
):
    """
    Execute one agent action.
    Body: {"tool_name": "...", "tool_args": {...}}
    Returns observation, reward [0.0-1.0], done, info.
    """
    try:
        return env.step(session_id, action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=StateResult)
def state(session_id: str = Query(..., description="Session ID from /reset")):
    """Return current state snapshot for a session."""
    try:
        return env.get_state(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# Catch-all JSON error handler
@app.exception_handler(Exception)
async def generic_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )