"""
Ekstrakcja wszystkich metryk (val + test) oraz czasu treningu dla
wszystkich wytrenowanych modeli z MLflow, do przeglądu w konsoli
i/lub eksportu do CSV - gotowe do wstawienia jako tabela w rozdziale 4.
"""

import mlflow
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def extract_metrics(output_csv: str = "reports/metrics_summary.csv") -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Nie znaleziono eksperymentu: {MLFLOW_EXPERIMENT_NAME}")

    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if "metrics.test_accuracy" not in runs_df.columns:
        raise RuntimeError("Brak zalogowanych metryk 'test_accuracy' w żadnym runie.")

    training_runs = runs_df[runs_df["metrics.test_accuracy"].notna()].copy()

    # Czas treningu: różnica end_time - start_time (automatycznie
    # zapisywane przez MLflow dla każdego runu, w milisekundach -> sekundy)
    training_runs["training_time_sec"] = (
        (training_runs["end_time"] - training_runs["start_time"])
        .dt.total_seconds()
    )

    metric_cols = [c for c in training_runs.columns if c.startswith("metrics.")]
    param_cols = ["params.dataset", "params.model_name"]

    columns_to_keep = param_cols + sorted(metric_cols) + ["training_time_sec"]
    result = training_runs[columns_to_keep].copy()

    result.columns = [c.replace("params.", "").replace("metrics.", "") for c in result.columns]

    result = result.sort_values(
        by=["dataset", "test_accuracy"], ascending=[True, False]
    ).reset_index(drop=True)

    print(f"Znaleziono {len(result)} runów treningowych.\n")
    print(result.to_string(index=False))

    if output_csv:
        from pathlib import Path
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"\nZapisano: {output_csv}")

    return result


def extract_training_time_summary(output_csv: str = "reports/training_time_summary.csv") -> pd.DataFrame:
    """Osobna, zwięzła tabela: dataset x model x czas treningu (sekundy) -
    gotowa do wstawienia jako tabela/wykres w rozdziale 4 (analiza
    kosztu obliczeniowego, wspiera hipotezę H3)."""
    full_df = extract_metrics(output_csv=None)

    summary = full_df[["dataset", "model_name", "training_time_sec"]].copy()
    summary = summary.sort_values(
        by=["dataset", "training_time_sec"], ascending=[True, True]
    ).reset_index(drop=True)

    print("\n=== Czas treningu (sekundy) - dataset x model ===")
    print(summary.to_string(index=False))

    if output_csv:
        from pathlib import Path
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)
        print(f"\nZapisano: {output_csv}")

    return summary


def extract_best_per_dataset(output_csv: str = "reports/best_models_summary.csv") -> pd.DataFrame:
    """Najlepszy model per dataset (wg test_roc_auc) - gotowa tabela
    do rozdziału 4/5."""
    full_df = extract_metrics(output_csv=None)

    if "test_roc_auc" not in full_df.columns:
        raise RuntimeError("Brak metryki 'test_roc_auc' do wyznaczenia najlepszych modeli.")

    best = (
        full_df.sort_values("test_roc_auc", ascending=False)
        .groupby("dataset", as_index=False)
        .first()
    )

    print("\n=== Najlepszy model per dataset (wg test_roc_auc) ===")
    print(best.to_string(index=False))

    if output_csv:
        from pathlib import Path
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        best.to_csv(output_csv, index=False)
        print(f"\nZapisano: {output_csv}")

    return best


if __name__ == "__main__":
    print("=== Wszystkie metryki + czas treningu (15 kombinacji) ===\n")
    extract_metrics()

    print("\n")
    extract_training_time_summary()

    print("\n")
    extract_best_per_dataset()
