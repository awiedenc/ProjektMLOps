import argparse
import skops.io as sio
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE
)
from src.dataset_configs import DATASET_CONFIGS
from src.data_loader import load_full_dataframe
from src.evaluate import evaluate_classification
from src.model_factory import get_models
from src.preprocessing import build_preprocessor
from src.utils import ensure_dir


def train_dataset(dataset_name: str):
    ensure_dir(MODELS_DIR)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    config = DATASET_CONFIGS[dataset_name]
    df = load_full_dataframe(dataset_name)

    target_col = config["target_col"]
    x = df.drop(columns=[target_col])
    y = df[target_col]

    # 60/20/20 split
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )

    print(f"Dataset: {dataset_name}")
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    preprocessor = build_preprocessor(
        config["numeric_features"],
        config["categorical_features"]
    )

    models = get_models()

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        with mlflow.start_run(run_name=f"{dataset_name}_{model_name}"):
            pipeline.fit(x_train, y_train)

            y_val_pred = pipeline.predict(x_val)
            y_val_proba = pipeline.predict_proba(x_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
            val_metrics = evaluate_classification(y_val, y_val_pred, y_val_proba)

            y_test_pred = pipeline.predict(x_test)
            y_test_proba = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            test_metrics = evaluate_classification(y_test, y_test_pred, y_test_proba)

            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target_col", target_col)

            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model"
            )

            model_path = MODELS_DIR / f"{dataset_name}_{model_name}.skops"
            sio.dump(pipeline, model_path)

            print(f"Zapisano model: {model_path}")
            print(f"Val metrics: {val_metrics}")
            print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["diabetes", "heart", "breast_cancer"]
    )
    args = parser.parse_args()

    train_dataset(args.dataset)