import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.sklearn
import os

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run():
        # Log all configs to MLflow
        mlflow.log_params({
            "dataset": cfg.dataset.name,
            "batch_size": cfg.dataset.batch_size,
            "model": cfg.model.type,
            "epochs": cfg.training.epochs,
            "lr": cfg.training.learning_rate,
            "optimizer": cfg.training.optimizer
        })

        # Dummy training loop (replace with real training code)
        for epoch in range(cfg.training.epochs):
            acc = 0.8 + epoch * 0.02
            loss = 1.0 / (epoch + 1)

            # Log metrics
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("loss", loss, step=epoch)

        # Example: save model
        model_path = os.path.join(os.getcwd(), "model.pkl")
        with open(model_path, "w") as f:
            f.write("dummy model")

        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
