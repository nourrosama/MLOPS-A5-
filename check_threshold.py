import os
import sys
import mlflow

THRESHOLD = 0.85

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# ── Read Run ID ───────────────────────────────────────────────────────────────
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# ── Fetch metrics from MLflow ─────────────────────────────────────────────────
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: 'accuracy' metric not found in MLflow run.")
    sys.exit(1)

print(f"Accuracy: {accuracy:.4f}  (threshold: {THRESHOLD})")

if accuracy < THRESHOLD:
    print(f"FAIL — accuracy {accuracy:.4f} is below threshold {THRESHOLD}.")
    sys.exit(1)

print(f"PASS — accuracy {accuracy:.4f} meets threshold {THRESHOLD}.")
