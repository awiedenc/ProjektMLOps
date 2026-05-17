from src.data_loader import load_dataset
from src.config import MODELS_DIR
import skops.io as sio
import pandas as pd

x, y, config = load_dataset("breast_cancer")
print(f"Dataset: {config['file_path']}")

sample = x.iloc[0].to_dict()
print("Sample input:", sample)

model = sio.load(MODELS_DIR / "breast_cancer_logistic_regression.skops", True)

pred = model.predict(pd.DataFrame([sample]))[0]
print("Predykcja modelu:", pred)
print("Surowy target y[0]:", y.iloc[0])