
import pandas as pd
import skops.io as sio
from src.config import MODELS_DIR

def load_model(model_filename: str):
    model_path = MODELS_DIR / model_filename
    unknown_types = sio.get_untrusted_types(file=model_path)
    return sio.load(model_path, trusted=unknown_types)


def _rename_breast_columns(payload: dict) -> dict:
    mapping = {
        "concave_points_mean": "concave points_mean",
        "concave_points_se": "concave points_se",
        "concave_points_worst": "concave points_worst",
    }
    return {mapping.get(k, k): v for k, v in payload.items()}


def predict_with_model(model_filename: str, payload: dict, dataset_name: str):
    model = load_model(model_filename)
    if dataset_name == "breast_cancer":
        payload = _rename_breast_columns(payload)
    df = pd.DataFrame([payload])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else None
    return {"prediction": prediction, "probability": probability}