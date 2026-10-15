import mlflow
from activity_ml.config import HARConfig
from activity_ml.classifier import train_from_config

def run_experiment(config_path: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("HAR_BiLSTM_Experiments")

    config = HARConfig.from_json(config_path)

    with mlflow.start_run():
        mlflow.log_params(
            {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "window_size": config.window_size,
            "loss_type": config.loss_type,
            "seed": config.seed,
            }
        )
        mlflow.log_dict(config.to_dict(), "training_config.json")

        def log_epoch_metrics(record):
            epoch = int(record['epoch'])
            mlflow.log_metric("train_loss", record["train_loss"], step=epoch)
            mlflow.log_metric("validation_loss", record["validation_loss"], step=epoch)
            mlflow.log_metric("train_accuracy", record["train_accuracy"], step=epoch)
            mlflow.log_metric("validation_accuracy", record["validation_accuracy"], step=epoch)
            train_loss = record["train_loss"]
            val_loss = record["validation_loss"]
            train_acc = record["train_accuracy"]
            val_acc = record["validation_accuracy"]
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}",
                flush=True
                )

        result = train_from_config(config, progress=log_epoch_metrics)

        metrics = result["test_metrics"]
        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("test_macro_f1", metrics["macro_avg"]["f1-score"])
        mlflow.log_metric("test_weighted_f1", metrics["weighted_avg"]["f1-score"])

        mlflow.log_artifact(result["checkpoint_path"], artifact_path="model")

        mlflow.log_artifact(result["metrics_path"], artifact_path="evaluation")
        mlflow.log_artifact(result["predictions_path"], artifact_path="evaluation")

        for figure_path in result["figure_paths"].values():
            mlflow.log_artifact(figure_path, artifact_path="figures")

        print("Experiment completed successfully!")

if __name__ == "__main__":
    run_experiment("configs/example_config.json")
