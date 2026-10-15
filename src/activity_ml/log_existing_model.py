import mlflow
from activity_ml.classifier import load_checkpoint


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("HAR_BiLSTM_Experiments")

checkpoint_path = "artifacts/activity_ml/best_model.pt"
checkpoint = load_checkpoint(checkpoint_path)

with mlflow.start_run(run_name="Existing_Baseline_Model"):
    mlflow.log_dict(checkpoint["training_config"],"training_config.json")
    mlflow.log_params(checkpoint["model_config"])
    metrics = checkpoint["metrics"]
    mlflow.log_metric("test_accuracy", metrics["accuracy"])
    mlflow.log_metric("test_macro_f1", metrics["macro_avg"]["f1-score"])
    mlflow.log_artifact(checkpoint_path, artifact_path="model")

print("Existing model logged successfully!")