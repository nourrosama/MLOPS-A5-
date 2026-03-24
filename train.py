import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("iris-classifier")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/iris.csv")

# 🔻 Drop most informative features (THIS MATTERS A LOT)
X = df.drop(["species", "petal_length", "petal_width"], axis=1)
y = df["species"]

# First split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔻 Keep ONLY 20% of training data
X_train, _, y_train, _ = train_test_split(
    X_train, y_train, test_size=0.8, random_state=42
)

# ── Train ─────────────────────────────────────────────────────────────────────
with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=50, C=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 50)
    mlflow.log_param("C", 0.1)
    mlflow.log_param("feature_reduction", "removed petal features")
    mlflow.log_param("train_size", "20% of training data")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id
    print(f"Run ID  : {run_id}")
    print(f"Accuracy: {accuracy:.4f}")

# ── Export Run ID ──────────────────────────────────────────────────────────────
with open("model_info.txt", "w") as f:
    f.write(run_id)

print("model_info.txt written.")