"""
Inference Script — Shipment Tracking Agent Environment
======================================================

Runs a baseline LLM agent against all 3 tasks and emits mandatory stdout logs.

MANDATORY ENV VARS:
  API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model name    (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key
  ENV_BASE_URL   Env server    (default: http://localhost:7860)

STDOUT FORMAT (strictly enforced):
  [START] task=<name> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────── #
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "sk-placeholder")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK               = "shipment-tracking-env"
MAX_STEPS_PER_TASK      = {"task1": 6, "task2": 9, "task3": 13}
SUCCESS_SCORE_THRESHOLD = 0.7
TEMPERATURE             = 0.2       # low for deterministic tool selection
MAX_LLM_TOKENS          = 300

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert logistics support agent resolving shipment exceptions.
You interact with a tool-based environment. Each turn you must call exactly one tool.

TOOLS and when to use them:
- get_shipment_status   → ALWAYS call this first to understand the situation
- notify_customer       → send a detailed, meaningful notification (min 20 chars)
- check_carrier_sla     → verify if the carrier has breached their SLA
- rebook_delivery       → reschedule a failed delivery (needs shipment_id + preferred_date)
- file_carrier_claim    → file official claim for lost package (needs shipment_id, carrier, reason)
- issue_refund          → issue refund for the EXACT order value shown in shipment details
- reship_order          → create replacement shipment (only after claim/refund)
- close_ticket          → ALWAYS call last with detailed resolution_notes (min 20 chars)

RESPONSE FORMAT — you must respond with ONLY valid JSON, nothing else:
{"tool_name": "<tool>", "tool_args": {"key": "value"}}

STRATEGY:
  task1 (delay):         get_status → notify_customer → close_ticket
  task2 (failed):        get_status → check_sla → rebook → notify → close
  task3 (lost):          get_status → check_sla → file_claim → issue_refund → reship → notify → close

Use exact IDs (shipment_id, order_id, customer_id) from the observation.
For issue_refund, use the EXACT 'order_value' from get_shipment_status output.
""").strip()


# ── Logging (mandatory stdout format) ─────────────────────────────────────── #

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise action: no newlines, truncate to 200 chars
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ──────────────────────────────────────────────────────────────── #

def get_model_action(
    client: OpenAI,
    step: int,
    obs: dict,
    last_reward: float,
    history: List[str],
) -> str:
    """Call the LLM and return raw JSON string for the next tool call."""
    history_block = "\n".join(history[-5:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Step {step} | Last reward: {last_reward:.2f}

        Current observation:
        {json.dumps(obs, indent=2)}

        Recent history:
        {history_block}

        What is your next tool call? Respond with JSON only.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_LLM_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if model wraps JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return text.strip() if text.strip() else fallback_action(step, obs)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return fallback_action(step, obs)


def fallback_action(step: int, obs: dict) -> str:
    """Rule-based fallback if LLM call fails."""
    tools = obs.get("available_tools", [])
    checked = False
    if step == 1 and "get_shipment_status" in tools:
        return json.dumps({"tool_name": "get_shipment_status", "tool_args": {}})
    if "close_ticket" in tools:
        return json.dumps({"tool_name": "close_ticket", "tool_args": {"resolution_notes": "Fallback closure — issue investigated."}})
    return json.dumps({"tool_name": tools[0] if tools else "get_shipment_status", "tool_args": {}})


# ── Episode runner ─────────────────────────────────────────────────────────── #

async def run_episode(
    client: OpenAI,
    http: httpx.AsyncClient,
    task_id: str,
) -> None:
    """Run one full episode for a single task."""
    max_steps = MAX_STEPS_PER_TASK[task_id]
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False
    history: List[str] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ── #
        resp = await http.post("/reset", params={"task_id": task_id})
        resp.raise_for_status()
        reset_data = resp.json()
        session_id = reset_data["session_id"]
        obs        = reset_data["observation"]
        last_reward = 0.0
        done        = False

        # ── Steps ── #
        for step in range(1, max_steps + 1):
            if done:
                break

            raw_action = get_model_action(client, step, obs, last_reward, history)

            # Parse action safely
            error: Optional[str] = None
            try:
                action_dict = json.loads(raw_action)
            except json.JSONDecodeError as je:
                error = f"JSON parse error: {je}"
                action_dict = {"tool_name": "get_shipment_status", "tool_args": {}}

            try:
                step_resp = await http.post(
                    "/step",
                    params={"session_id": session_id},
                    json=action_dict,
                )
                step_resp.raise_for_status()
                result      = step_resp.json()
                obs         = result["observation"]
                reward      = float(result.get("reward", 0.0))
                done        = bool(result.get("done", False))
            except Exception as exc:
                error  = str(exc)
                reward = 0.0
                done   = False

            rewards.append(reward)
            steps_taken  = step
            last_reward  = reward

            action_str = json.dumps(action_dict).replace('"', "'")
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {json.dumps(action_dict)[:100]} → reward {reward:.2f}")

    finally:
        # Score = final reward (already cumulative from grader), clamped
        final_reward = rewards[-1] if rewards else 0.0
        score   = round(min(max(final_reward, 0.0), 1.0), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────── #

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=60) as http:
        # Health check
        try:
            hc = await http.get("/health")
            hc.raise_for_status()
            print(f"[DEBUG] Environment healthy: {hc.json()}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] Health check failed: {exc}", flush=True)

        # Run all 3 tasks sequentially
        for task_id in ("task1", "task2", "task3"):
            await run_episode(client, http, task_id)
            print("", flush=True)   # blank line between tasks


if __name__ == "__main__":
    asyncio.run(main())