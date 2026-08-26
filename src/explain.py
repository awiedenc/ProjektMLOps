"""
Moduł do generowania analizy interpretowalności SHAP
dla wytrenowanych modeli wczytywanych z plików .skops.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
import skops.io as sio
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
)
from src.data_loader import load_full_dataframe
from src.dataset_configs import DATASET_CONFIGS


TREE_MODELS = ("random_forest", "xgboost")
LINEAR_MODELS = ("logistic_regression",)
# Pozostałe modele (mlp_shallow, mlp_deep) -> KernelExplainer


def to_dense(x):
    """Konwertuje macierz sparse, np. po OneHotEncoder, do dense array."""
    if issparse(x):
        return x.toarray()
    return x


def load_pipeline(dataset_name: str, model_name: str):
    """
    Wczytuje wytrenowany pipeline z pliku .skops.

    Pliki wykorzystywane w projekcie pochodzą z kontrolowanego procesu
    treningowego realizowanego przez train.py.
    """
    model_path = MODELS_DIR / f"{dataset_name}_{model_name}.skops"

    untrusted_types = sio.get_untrusted_types(file=model_path)

    return sio.load(
        model_path,
        trusted=untrusted_types,
    )


def get_feature_names(pipeline) -> list:
    """Pobiera nazwy cech po preprocessingu, np. po OneHotEncoder."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def get_train_test_data(dataset_name: str):
    """
    Odtwarza ten sam podział 60/20/20, który jest wykorzystywany
    podczas treningu modeli w train.py.

    Zwraca:
        x_train, x_test
    """
    config = DATASET_CONFIGS[dataset_name]
    df = load_full_dataframe(dataset_name)

    target_col = config["target_col"]

    x = df.drop(columns=[target_col])
    y = df[target_col]

    # Pierwszy podział: 80% train+validation / 20% test
    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Drugi podział: 60% train / 20% validation
    x_train, _, y_train, _ = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return x_train, x_test


def get_explainer(
    pipeline,
    model_name: str,
    x_background: np.ndarray,
):
    """
    Wybiera odpowiedni typ SHAP explainer w zależności od modelu.

    Dla Logistic Regression i MLP wykorzystywana jest próbka
    zbioru treningowego jako dane referencyjne (background).
    """
    classifier = pipeline.named_steps["classifier"]

    if model_name in TREE_MODELS:
        return shap.TreeExplainer(classifier)

    background_sample = shap.sample(
        x_background,
        min(100, len(x_background)),
        random_state=RANDOM_STATE,
    )

    if model_name in LINEAR_MODELS:
        return shap.LinearExplainer(
            classifier,
            background_sample,
        )

    # mlp_shallow, mlp_deep
    return shap.KernelExplainer(
        classifier.predict_proba,
        background_sample,
    )


def compute_shap_values(
    dataset_name: str,
    model_name: str,
    n_samples: int = 200,
):
    """
    Oblicza wartości SHAP dla obserwacji ze zbioru testowego.

    Dane treningowe są wykorzystywane jako źródło background
    dla explainerów, które wymagają danych referencyjnych.

    Zwraca:
        shap_values,
        x_transformed,
        feature_names,
        expected_value
    """
    x_train, x_test = get_train_test_data(dataset_name)

    # Analiza SHAP wykonywana jest na zbiorze testowym.
    # Reset indeksu sprawia, że sample_index=0 oznacza
    # pierwszą obserwację analizowanej części testowej.
    x_sample = (
        x_test
        .reset_index(drop=True)
        .iloc[:min(n_samples, len(x_test))]
    )

    pipeline = load_pipeline(dataset_name, model_name)
    preprocessor = pipeline.named_steps["preprocessor"]

    x_background = to_dense(
        preprocessor.transform(x_train)
    )

    x_transformed = to_dense(
        preprocessor.transform(x_sample)
    )

    feature_names = get_feature_names(pipeline)

    explainer = get_explainer(
        pipeline,
        model_name,
        x_background,
    )

    shap_values = explainer.shap_values(x_transformed)

    return (
        shap_values,
        x_transformed,
        feature_names,
        explainer.expected_value,
    )


def _extract_positive_class(
    shap_values,
    expected_value,
):
    """
    Wybiera wartości SHAP odpowiadające klasie pozytywnej.

    Różne explainery mogą zwracać:
    - macierz 2D (samples, features),
    - listę [class_0, class_1],
    - tablicę 3D (samples, features, classes).

    Analogicznie expected_value może być skalarem
    albo zawierać osobną wartość dla każdej klasy.
    """
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]

    elif (
        isinstance(shap_values, np.ndarray)
        and shap_values.ndim == 3
    ):
        shap_values_pos = shap_values[:, :, 1]

    else:
        shap_values_pos = shap_values

    expected_array = np.asarray(expected_value)

    if expected_array.ndim > 0 and expected_array.size > 1:
        base_value = float(expected_array.reshape(-1)[1])
    else:
        base_value = float(expected_array.reshape(-1)[0])

    return shap_values_pos, base_value


# ---------------------------------------------------------
# Globalna ważność cech (summary plot)
# ---------------------------------------------------------

def plot_global_importance(
    dataset_name: str,
    model_name: str,
    output_dir: str = "reports/shap",
):
    """Generuje i zapisuje wykres globalnej ważności cech."""
    (
        shap_values,
        x_transformed,
        feature_names,
        expected_value,
    ) = compute_shap_values(
        dataset_name,
        model_name,
    )

    shap_values_pos, _ = _extract_positive_class(
        shap_values,
        expected_value,
    )

    plt.figure()

    shap.summary_plot(
        shap_values_pos,
        x_transformed,
        feature_names=feature_names,
        show=False,
    )

    plt.title(
        f"SHAP - {dataset_name} - {model_name}"
    )

    plt.tight_layout()

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = (
        f"{output_dir}/"
        f"{dataset_name}_{model_name}_shap_summary.png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Zapisano: {save_path}")

    return save_path


# ---------------------------------------------------------
# Integracja z MLflow
# ---------------------------------------------------------

def log_shap_to_mlflow(
    dataset_name: str,
    model_name: str,
):
    """Generuje wykres SHAP i zapisuje go jako artefakt MLflow."""
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=f"{dataset_name}_{model_name}_shap"
    ):
        save_path = plot_global_importance(
            dataset_name,
            model_name,
        )

        mlflow.log_artifact(
            save_path,
            artifact_path="shap",
        )

        mlflow.log_param(
            "dataset",
            dataset_name,
        )

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "explanation_type",
            "shap_global",
        )


# ---------------------------------------------------------
# Lokalna analiza (wyjaśnienie jednej predykcji)
# ---------------------------------------------------------

def plot_local_explanation(
    dataset_name: str,
    model_name: str,
    sample_index: int = 0,
    output_dir: str = "reports/shap",
):
    """
    Generuje lokalne wyjaśnienie pojedynczej obserwacji
    ze zbioru testowego w postaci waterfall plot.
    """
    (
        shap_values,
        x_transformed,
        feature_names,
        expected_value,
    ) = compute_shap_values(
        dataset_name,
        model_name,
    )

    shap_values_pos, base_value = _extract_positive_class(
        shap_values,
        expected_value,
    )

    explanation = shap.Explanation(
        values=shap_values_pos[sample_index],
        base_values=base_value,
        data=x_transformed[sample_index],
        feature_names=feature_names,
    )

    plt.figure()

    shap.plots.waterfall(
        explanation,
        show=False,
    )

    plt.title(
        f"SHAP local - {dataset_name} - "
        f"{model_name} - sample {sample_index}"
    )

    plt.tight_layout()

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = (
        f"{output_dir}/"
        f"{dataset_name}_{model_name}_"
        f"shap_local_{sample_index}.png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Zapisano: {save_path}")

    return save_path


if __name__ == "__main__":
    datasets = [
        "diabetes",
        "heart",
        "breast_cancer",
    ]

    models = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "mlp_shallow",
        "mlp_deep",
    ]

    combinations = [
        (dataset, model)
        for dataset in datasets
        for model in models
    ]

    for dataset_name, model_name in combinations:
        print(
            f"\n=== SHAP: "
            f"{dataset_name} / {model_name} ==="
        )

        try:
            plot_global_importance(
                dataset_name,
                model_name,
            )

            plot_local_explanation(
                dataset_name,
                model_name,
                sample_index=0,
            )

            log_shap_to_mlflow(
                dataset_name,
                model_name,
            )

        except Exception as e:
            print(
                f"BŁĄD dla "
                f"{dataset_name}/{model_name}: {e}"
            )
            continue