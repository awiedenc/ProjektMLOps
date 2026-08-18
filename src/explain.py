"""
Moduł do generowania analizy interpretowalności SHAP
dla wytrenowanych modeli (wczytywanych z .skops).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
import skops.io as sio
from scipy.sparse import issparse

from src.config import MODELS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.data_loader import load_full_dataframe
from src.dataset_configs import DATASET_CONFIGS

TREE_MODELS = ("random_forest", "xgboost")
LINEAR_MODELS = ("logistic_regression",)
# wszystko inne (mlp_shallow, mlp_deep) -> KernelExplainer


def to_dense(x):
    """Konwertuje sparse matrix (z OneHotEncoder) na dense array."""
    if issparse(x):
        return x.toarray()
    return x


def load_pipeline(dataset_name: str, model_name: str):
    """Wczytuje wytrenowany pipeline z pliku .skops.
    Dynamicznie wykrywa i zatwierdza wymagane typy - bezpieczne,
    bo pliki pochodzą z naszego własnego, kontrolowanego treningu
    (train.py), a nie z niezaufanego, zewnętrznego źródła."""
    model_path = MODELS_DIR / f"{dataset_name}_{model_name}.skops"
    untrusted_types = sio.get_untrusted_types(file=model_path)
    return sio.load(model_path, trusted=untrusted_types)


def get_feature_names(pipeline) -> list:
    """Wyciąga poprawne nazwy cech PO preprocessingu (np. po one-hot)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())

def get_explainer(pipeline, model_name: str, x_transformed: np.ndarray):
    """Wybiera odpowiedni typ SHAP explainer w zależności od modelu."""
    classifier = pipeline.named_steps["classifier"]

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(classifier)
    elif model_name in LINEAR_MODELS:
        explainer = shap.LinearExplainer(classifier, x_transformed)
    else:  # mlp_shallow, mlp_deep
        background_sample = shap.sample(x_transformed, min(100, len(x_transformed)))
        explainer = shap.KernelExplainer(classifier.predict_proba, background_sample)

    return explainer


def compute_shap_values(dataset_name: str, model_name: str, n_samples: int = 200):
    """
    Liczy wartości SHAP dla danego modelu i datasetu.
    Zwraca (shap_values, x_transformed, feature_names).
    """
    config = DATASET_CONFIGS[dataset_name]
    df = load_full_dataframe(dataset_name)
    target_col = config["target_col"]

    x = df.drop(columns=[target_col])
    x_sample = x.sample(n=min(n_samples, len(x)), random_state=42).reset_index(drop=True)

    pipeline = load_pipeline(dataset_name, model_name)
    preprocessor = pipeline.named_steps["preprocessor"]

    x_transformed = to_dense(preprocessor.transform(x_sample))
    feature_names = get_feature_names(pipeline)

    explainer = get_explainer(pipeline, model_name, x_transformed)
    shap_values = explainer.shap_values(x_transformed)

    return shap_values, x_transformed, feature_names


def _extract_positive_class(shap_values):
    """Dla klasyfikacji binarnej niektóre explainery zwracają listę [class0, class1]
    lub tablicę 3D (samples, features, classes)."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


# ---------------------------------------------------------
# Globalna ważność cech (summary plot)
# ---------------------------------------------------------

def plot_global_importance(dataset_name: str, model_name: str,
                            output_dir: str = "reports/shap"):
    """Generuje i zapisuje wykres globalnej ważności cech (summary plot)."""
    shap_values, x_transformed, feature_names = compute_shap_values(
        dataset_name, model_name
    )
    shap_values_pos = _extract_positive_class(shap_values)

    plt.figure()
    shap.summary_plot(
        shap_values_pos, x_transformed, feature_names=feature_names, show=False
    )
    plt.title(f"SHAP - {dataset_name} - {model_name}")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/{dataset_name}_{model_name}_shap_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {save_path}")

    return save_path


# ---------------------------------------------------------
# Integracja z MLflow
# ---------------------------------------------------------

def log_shap_to_mlflow(dataset_name: str, model_name: str):
    """Liczy SHAP i loguje wykres jako artefakt do MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{dataset_name}_{model_name}_shap"):
        save_path = plot_global_importance(dataset_name, model_name)
        mlflow.log_artifact(save_path, artifact_path="shap")
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("explanation_type", "shap_global")


# ---------------------------------------------------------
# Lokalna analiza (wyjaśnienie jednej predykcji)
# ---------------------------------------------------------

def plot_local_explanation(dataset_name: str, model_name: str,
                            sample_index: int = 0,
                            output_dir: str = "reports/shap"):
    """Wyjaśnienie pojedynczej predykcji - waterfall plot."""
    shap_values, x_transformed, feature_names = compute_shap_values(
        dataset_name, model_name
    )
    shap_values_pos = _extract_positive_class(shap_values)

    explanation = shap.Explanation(
        values=shap_values_pos[sample_index],
        base_values=float(np.mean(shap_values_pos)),
        data=x_transformed[sample_index],
        feature_names=feature_names
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP local - {dataset_name} - {model_name} - sample {sample_index}")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/{dataset_name}_{model_name}_shap_local_{sample_index}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Zapisano: {save_path}")

    return save_path


if __name__ == "__main__":
    datasets = ["diabetes", "heart", "breast_cancer"]
    models = ["logistic_regression", "random_forest", "xgboost",
              "mlp_shallow", "mlp_deep"]

    combinations = [(d, m) for d in datasets for m in models]

    for dataset_name, model_name in combinations:
        print(f"\n=== SHAP: {dataset_name} / {model_name} ===")
        try:
            plot_global_importance(dataset_name, model_name)
            plot_local_explanation(dataset_name, model_name, sample_index=0)
            log_shap_to_mlflow(dataset_name, model_name)
        except Exception as e:
            print(f"BŁĄD dla {dataset_name}/{model_name}: {e}")
            continue
