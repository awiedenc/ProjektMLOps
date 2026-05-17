import pandas as pd

from src.dataset_configs import get_dataset_config


def transform_target(y: pd.Series, transform_type: str) -> pd.Series:
    if transform_type == "none":
        return y.astype(int)

    if transform_type == "heart_binary":
        return (pd.to_numeric(y, errors="coerce") > 0).astype(int)

    if transform_type == "breast_binary":
        return y.map({"B": 0, "M": 1}).astype(int)

    raise ValueError(f"Nieobsługiwany target_transform: {transform_type}")


def load_dataset(dataset_name: str, stage: str = "processed"):
    config = get_dataset_config(dataset_name, stage=stage)
    df = pd.read_csv(config["file_path"])

    df.columns = [col.strip().strip('"') for col in df.columns]

    if "" in df.columns:
        df = df.drop(columns=[""])

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    drop_cols = [col for col in config["drop_cols"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    target_col = config["target_col"]
    y = transform_target(df[target_col], config["target_transform"])
    x = df.drop(columns=[target_col])

    return x, y, config


def load_full_dataframe(dataset_name: str) -> pd.DataFrame:
    config = get_dataset_config(dataset_name, stage="processed")
    df = pd.read_csv(config["file_path"])
    df.columns = [col.strip().strip('"') for col in df.columns]

    if "" in df.columns:
        df = df.drop(columns=[""])

    return df