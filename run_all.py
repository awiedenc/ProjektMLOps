from src.train import train_dataset
from src.prepare_data import main as prepare_all
from src.explain import (
    plot_global_importance,
    plot_local_explanation,
    log_shap_to_mlflow,
)
from src.plots import log_plots_to_mlflow

DATASETS = ["diabetes", "heart", "breast_cancer"]
MODELS = ["logistic_regression", "random_forest", "xgboost",
          "mlp_shallow", "mlp_deep"]


def main():
    print("=== Krok 1: Prepare data ===")
    prepare_all()

    print("\n=== Krok 2: Train models ===")
    for dataset in DATASETS:
        print(f"\n--- Training {dataset} ---")
        train_dataset(dataset)

    print("\n=== Krok 3: SHAP (interpretowalność) ===")
    for dataset in DATASETS:
        for model_name in MODELS:
            print(f"\n--- SHAP: {dataset} / {model_name} ---")
            try:
                plot_global_importance(dataset, model_name)
                plot_local_explanation(dataset, model_name, sample_index=0)
                log_shap_to_mlflow(dataset, model_name)
            except Exception as e:
                print(f"BŁĄD SHAP dla {dataset}/{model_name}: {e}")
                continue

    print("\n=== Krok 4: ROC curve + Confusion matrix ===")
    for dataset in DATASETS:
        print(f"\n--- Plots: {dataset} ---")
        try:
            log_plots_to_mlflow(dataset, MODELS)
        except Exception as e:
            print(f"BŁĄD plots dla {dataset}: {e}")
            continue

    print("\n=== Pipeline zakończony ===")


if __name__ == "__main__":
    main()