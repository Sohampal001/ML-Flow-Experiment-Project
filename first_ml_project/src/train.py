import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.sklearn
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, ConfusionMatrixDisplay


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run():
        # === Log config parameters ===
        mlflow.log_params({
            "dataset": cfg.dataset.name,
            "batch_size": cfg.dataset.batch_size,
            "model": cfg.model.type,
            "epochs": cfg.training.epochs,
            "lr": cfg.training.learning_rate,
            "optimizer": cfg.training.optimizer,
            "n_estimators": cfg.training.n_estimators,
        })

        # === Load dataset ===
        if cfg.dataset.name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            feature_names = data.feature_names
        else:
            raise ValueError(f"Dataset {cfg.dataset.name} not supported yet!")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # === Train model ===
        if cfg.model.type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=cfg.training.n_estimators,
                random_state=42
            )
        else:
            raise ValueError(f"Model {cfg.model.type} not supported yet!")

        model.fit(X_train, y_train)

        # === Evaluate model ===
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("loss", loss)

        # === Log model directly to MLflow ===
        mlflow.sklearn.log_model(model, "model")

        # === Extra artifacts logging ===

        # Feature importance CSV
        if cfg.model.type == "random_forest":
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            })
            fi_path = "feature_importances.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

        # Confusion matrix plot
        cm_disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        print(f"✅ Training complete. Accuracy={acc:.4f}, Loss={loss:.4f}")
        print("Artifacts logged: model, feature_importances.csv, confusion_matrix.png")


if __name__ == "__main__":
    main()
