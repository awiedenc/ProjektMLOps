from pathlib import Path

import pandas as pd
import skops.io as sio

from src.config import MODELS_DIR


def load_model(model_filename: str):
    """
    Wczytuje pipeline zapisany w formacie .skops.

    Artefakty wykorzystywane przez API pochodzą
    z kontrolowanego procesu własnego treningu.
    """
    model_path = MODELS_DIR / model_filename

    untrusted_types = sio.get_untrusted_types(
        file=model_path
    )

    return sio.load(
        model_path,
        trusted=untrusted_types,
    )


def _rename_breast_columns(payload: dict) -> dict:
    """
    Przywraca nazwy wybranych kolumn Breast Cancer
    zgodne z nazwami używanymi w danych treningowych.
    """
    mapping = {
        "concave_points_mean": "concave points_mean",
        "concave_points_se": "concave points_se",
        "concave_points_worst": "concave points_worst",
    }

    return {
        mapping.get(key, key): value
        for key, value in payload.items()
    }


def _get_model_name(
    model_filename: str,
    dataset_name: str,
) -> str:
    """
    Wyodrębnia nazwę modelu z nazwy pliku.

    Przykład:
    diabetes_xgboost.skops -> xgboost
    """
    model_stem = Path(model_filename).stem

    prefix = f"{dataset_name}_"

    if model_stem.startswith(prefix):
        return model_stem[len(prefix):]

    return model_stem


def predict_with_model(
    model_filename: str,
    payload: dict,
    dataset_name: str,
):
    """
    Wykonuje predykcję przy użyciu zapisanego pipeline'u.

    Dane wejściowe są przekazywane jako surowe,
    nazwane cechy. Zapisany pipeline wykonuje
    preprocessing oraz właściwą predykcję modelu.
    """
    model = load_model(model_filename)

    if dataset_name == "breast_cancer":
        payload = _rename_breast_columns(
            payload
        )

    df = pd.DataFrame([payload])

    prediction = int(
        model.predict(df)[0]
    )

    probability = (
        float(
            model.predict_proba(df)[0][1]
        )
        if hasattr(model, "predict_proba")
        else None
    )

    model_name = _get_model_name(
        model_filename,
        dataset_name,
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "model_name": model_name,
        "dataset": dataset_name,
    }