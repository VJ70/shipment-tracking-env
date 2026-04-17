# Shipment Tracking Agent Environment

An OpenEnv-compliant environment where an AI agent resolves real-world
shipment exceptions — delays, failed deliveries, and lost packages.  
Delpoyed on HuggingFace: https://huggingface.co/spaces/VJ7-7/shipment-tracking-env

## Tasks
| ID | Name | Difficulty | Max Steps | Tools Required |
|----|------|-----------|-----------|----------------|
| task1 | Delay Notification | Easy | 5 | get_shipment_status, notify_customer |
| task2 | Failed Delivery Rebook | Medium | 8 | + check_carrier_sla, rebook_delivery |
| task3 | Lost Package Resolution | Hard | 12 | + file_carrier_claim, issue_refund, reship_order |

## Reward Structure
Rewards are partial — the agent earns score for each correct step,
not just task completion. All rewards are in range [0, 1].

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Run Inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here
python inference.py
```
