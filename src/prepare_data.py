import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.dataset_configs import get_dataset_config
from src.utils import ensure_dir


def prepare_dataset(dataset_name: str):
    config = get_dataset_config(dataset_name, stage="raw")
    df = pd.read_csv(config["file_path"])

    print(f"\n=== Preparing dataset: {dataset_name} ===")
    print(f"Original shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

    # Clean column names
    df.columns = [col.strip().strip('"') for col in df.columns]

    if "" in df.columns:
        df = df.drop(columns=[""])

    raw_target_col = "num" if dataset_name == "heart" else config["target_col"]

    if raw_target_col not in df.columns:
        raise ValueError(
            f"Raw target column '{raw_target_col}' not found in dataset '{dataset_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"Raw target column: '{raw_target_col}'")

    drop_cols = [col for col in config.get("drop_cols", []) if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped columns: {drop_cols}")

    target_col = config["target_col"]  # 'target' dla processed output

    if dataset_name == "heart":
        df[target_col] = pd.to_numeric(df[raw_target_col], errors="coerce")

        invalid_before = df[target_col].isna().sum()
        if invalid_before > 0:
            print(f"Warning: {invalid_before} rows have invalid target → dropping")

        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)

        df[target_col] = (df[target_col] > 0).astype(int)
        df = df.drop(columns=[raw_target_col])

    elif dataset_name == "breast_cancer":
        df[target_col] = df[raw_target_col].astype(str).str.strip().map({"B": 0, "M": 1})

        invalid_before = df[target_col].isna().sum()
        if invalid_before > 0:
            print(f"Warning: {invalid_before} rows have invalid diagnosis → dropping")

        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)

    elif dataset_name == "diabetes":
        df[target_col] = pd.to_numeric(df[raw_target_col], errors="coerce")

        invalid_before = df[target_col].isna().sum()
        if invalid_before > 0:
            print(f"Warning: {invalid_before} rows have invalid diabetes → dropping")

        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)

        invalid_mask = ~df[target_col].isin([0, 1])
        if invalid_mask.any():
            print(f"Warning: Dropping {invalid_mask.sum()} rows with unexpected values")
            df = df.loc[~invalid_mask].copy()

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Clean object columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        if col != target_col:
            df[col] = df[col].astype(str).str.strip()

    # Final validation
    print(f"Final shape: {df.shape}")
    print(f"Final {target_col} dtype: {df[target_col].dtype}")
    print(f"Final unique {target_col}: {sorted(df[target_col].unique())}")
    print(f"Target balance:")
    print(df[target_col].value_counts())
    print(df[target_col].value_counts(normalize=True).round(3))

    ensure_dir(PROCESSED_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Saved: {output_path}")
    return output_path


def main():
    datasets = ["diabetes", "heart", "breast_cancer"]
    for dataset in datasets:
        try:
            prepare_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")


if __name__ == "__main__":
    main()