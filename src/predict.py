import skops.io as sio
import pandas as pd

from src.config import MODELS_DIR


def predict_from_dict(model_filename: str, input_data: dict):
    model_path = MODELS_DIR / model_filename
    model = sio.load(model_path, True)

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": int(prediction),
        "probability": probability
    }