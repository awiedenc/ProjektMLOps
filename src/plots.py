"""
Moduł do generowania wykresów ewaluacyjnych (ROC curve, confusion matrix)
dla wytrenowanych modeli (wczytywanych z .skops).

Odtwarza IDENTYCZNY podział train/val/test (60/20/20, stratified,
RANDOM_STATE) co train.py, żeby wykresy były w 100% zgodne
z metrykami już zalogowanymi do MLflow.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, RANDOM_STATE
from src.data_loader import load_full_dataframe
from src.dataset_configs import DATASET_CONFIGS
from src.explain import load_pipeline  # reuse: dynamiczny loader .skops

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost",
               "mlp_shallow", "mlp_deep"]

MODEL_COLORS = {
    "logistic_regression": "#41605E",
    "random_forest": "#FF6359",
    "xgboost": "#718886",
    "mlp_shallow": "#A0AFAF",
    "mlp_deep": "#2E4443",
}


def get_test_split(dataset_name: str):
    """Odtwarza IDENTYCZNY podział 60/20/20 co train.py, żeby zbiór
    testowy użyty tutaj zgadzał się z metrykami zalogowanymi w MLflow."""
    config = DATASET_CONFIGS[dataset_name]
    df = load_full_dataframe(dataset_name)

    target_col = config["target_col"]
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # x_temp/y_temp dalej dzielone są w train.py na train/val - tutaj
    # potrzebujemy tylko x_test/y_test, więc reszta jest ignorowana
    return x_test, y_test


def plot_roc_curve(dataset_name: str, model_names: list = None,
                    output_dir: str = "reports/plots") -> str:
    """Rysuje ROC curve dla wszystkich podanych modeli na jednym
    wykresie (per dataset), żeby ułatwić porównanie między modelami."""
    if model_names is None:
        model_names = MODEL_NAMES

    x_test, y_test = get_test_split(dataset_name)

    plt.figure(figsize=(7, 6))

    for model_name in model_names:
        try:
            pipeline = load_pipeline(dataset_name, model_name)
        except FileNotFoundError:
            print(f"Pominięto {model_name} (brak zapisanego modelu .skops)")
            continue

        y_proba = pipeline.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_value = auc(fpr, tpr)

        color = MODEL_COLORS.get(model_name, "#718886")
        plt.plot(fpr, tpr, color=color, lw=2,
                  label=f"{model_name} (AUC = {roc_auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="#A0AFAF", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/{dataset_name}_roc_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {save_path}")

    return save_path


def plot_confusion_matrix(dataset_name: str, model_name: str,
                           output_dir: str = "reports/plots") -> str:
    """Confusion matrix dla jednej kombinacji dataset/model."""
    x_test, y_test = get_test_split(dataset_name)
    pipeline = load_pipeline(dataset_name, model_name)

    y_pred = pipeline.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Confusion Matrix - {dataset_name} - {model_name}")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/{dataset_name}_{model_name}_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {save_path}")

    return save_path


def log_plots_to_mlflow(dataset_name: str, model_names: list = None):
    """Generuje ROC curve (per dataset) + confusion matrix (per model)
    i loguje jako artefakty do MLflow - wzorem log_shap_to_mlflow."""
    if model_names is None:
        model_names = MODEL_NAMES

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{dataset_name}_evaluation_plots"):
        roc_path = plot_roc_curve(dataset_name, model_names)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        for model_name in model_names:
            try:
                cm_path = plot_confusion_matrix(dataset_name, model_name)
                mlflow.log_artifact(cm_path, artifact_path="plots")
            except FileNotFoundError:
                continue

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("plot_type", "roc_and_confusion_matrix")


if __name__ == "__main__":
    datasets = ["diabetes", "heart", "breast_cancer"]

    for dataset_name in datasets:
        print(f"\n=== Plots: {dataset_name} ===")
        log_plots_to_mlflow(dataset_name)
