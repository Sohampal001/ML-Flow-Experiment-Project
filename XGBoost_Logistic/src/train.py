import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(run_name=cfg.model.type):
        
        model_type = cfg.model.type.lower()
        model = None

        if model_type == "xgboost":
            mlflow.log_params({
                "n_estimators": cfg.params.n_estimators,
                "max_depth": cfg.params.max_depth,
                "learning_rate": cfg.params.learning_rate,
                "subsample": cfg.params.subsample,
                "colsample_bytree": cfg.params.colsample_bytree,
                "eval_metric": cfg.params.eval_metric,
            })
            model = xgb.XGBClassifier(
                n_estimators=cfg.params.n_estimators,
                max_depth=cfg.params.max_depth,
                learning_rate=cfg.params.learning_rate,
                subsample=cfg.params.subsample,
                colsample_bytree=cfg.params.colsample_bytree,
                use_label_encoder=False,
                eval_metric=cfg.params.eval_metric,
            )

        elif model_type == "logistic_regression":
            mlflow.log_params({
                "penalty": cfg.params.penalty,
                "l1_ratio": cfg.params.l1_ratio,
                "C": cfg.params.c,
                "solver": cfg.params.solver,
                "max_iter": cfg.params.max_iter,
            })
            model = LogisticRegression(
                C=cfg.params.c,
                penalty=cfg.params.penalty,
                l1_ratio=cfg.params.l1_ratio,
                solver=cfg.params.solver,
                max_iter=cfg.params.max_iter,
            )

        elif model_type == "random_forest":
            mlflow.log_params({
                "n_estimators": cfg.params.n_estimators,
                "max_depth": cfg.params.max_depth,
                "random_state": cfg.params.random_state,
            })
            model = RandomForestClassifier(
                n_estimators=cfg.params.n_estimators,
                max_depth=cfg.params.max_depth,
                random_state=cfg.params.random_state,
            )

        elif model_type == "svm":
            mlflow.log_params({
                "kernel": cfg.params.kernel,
                "C": cfg.params.c,
                "gamma": cfg.params.gamma,
            })
            model = SVC(
                kernel=cfg.params.kernel,
                C=cfg.params.c,
                gamma=cfg.params.gamma,
                probability=True
            )

        else:
            raise ValueError(f"Unsupported model type: {cfg.model.type}")

    
        if hasattr(cfg.dataset, "train_path") and hasattr(cfg.dataset, "test_path"):
            df_train = pd.read_csv(cfg.dataset.train_path)
            df_test = pd.read_csv(cfg.dataset.test_path)

            X_train = df_train.drop(columns=[cfg.dataset.target])
            y_train = df_train[cfg.dataset.target]

            X_test = df_test.drop(columns=[cfg.dataset.target])
            y_test = df_test[cfg.dataset.target]
        else:
            df = pd.read_csv(cfg.dataset.processed_path)
            X = df.drop(columns=[cfg.dataset.target])
            y = df[cfg.dataset.target]

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=cfg.dataset.test_size,
                random_state=cfg.dataset.random_state
            )

        print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loss = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            loss = log_loss(y_test, y_proba)
            mlflow.log_metric("log_loss", loss)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

    
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            fi_path = "feature_importances.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, artifact_path="feature_importance")

            plt.figure(figsize=(10, 6))
            sns.barplot(x="importance", y="feature", data=fi_df)
            plt.title(f"Feature Importances ({cfg.model.type})")
            mlflow.log_figure(plt.gcf(), "feature_importance/feature_importances.png")
            plt.close()

        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"Confusion Matrix ({cfg.model.type})")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path)
        plt.close()

        print(f"Training complete. Model={cfg.model.type}")
        print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, LogLoss={loss}")


if __name__ == "__main__":
    main()
