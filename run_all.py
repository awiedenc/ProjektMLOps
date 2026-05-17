from src.train import train_dataset
from src.prepare_data import main as prepare_all

def main():
    print("=== Krok 1: Prepare data ===")
    prepare_all()

    print("\n=== Krok 2: Train models ===")
    for dataset in ["diabetes", "heart", "breast_cancer"]:
        print(f"\n--- Training {dataset} ---")
        train_dataset(dataset)


if __name__ == "__main__":
    main()