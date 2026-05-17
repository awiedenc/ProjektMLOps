from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

DATASET_CONFIGS = {
    "diabetes": {
        "raw_path": RAW_DATA_DIR / "diabetes_prediction.csv",
        "processed_path": PROCESSED_DATA_DIR / "diabetes_clean.csv",
        "target_col": "diabetes",
        "drop_cols": [],
        "numeric_features": [
            "age",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ],
        "categorical_features": [
            "gender",
            "hypertension",
            "heart_disease",
            "smoking_history"
        ],
        "target_transform": "none"
    },
    "heart": {
        "raw_path": RAW_DATA_DIR / "heart_disease.csv",
        "processed_path": PROCESSED_DATA_DIR / "heart_clean.csv",
        "target_col": "target",
        "drop_cols": ["id", "dataset"],
        "numeric_features": [
            "age",
            "trestbps",
            "chol",
            "thalch",
            "oldpeak",
            "ca"
        ],
        "categorical_features": [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "thal"
        ],
        "target_transform": "heart_binary"
    },
    "breast_cancer": {
        "raw_path": RAW_DATA_DIR / "breast_cancer_wisconsin.csv",
        "processed_path": PROCESSED_DATA_DIR / "breast_cancer_clean.csv",
        "target_col": "diagnosis",
        "drop_cols": ["id"],
        "numeric_features": [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
            "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
            "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
            "symmetry_worst", "fractal_dimension_worst"
        ],
        "categorical_features": [],
        "target_transform": "breast_binary"
    }
}


def get_dataset_config(dataset_name: str, stage: str = "raw") -> dict:
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Nieznany dataset: {dataset_name}. Dostępne: {available}")

    config = DATASET_CONFIGS[dataset_name].copy()

    if stage == "raw":
        config["file_path"] = config["raw_path"]
    elif stage == "processed":
        config["file_path"] = config["processed_path"]
        config["target_transform"] = "none"  # processed już binarne
        config["drop_cols"] = []
    else:
        raise ValueError(f"Nieznany stage: {stage}")

    return config