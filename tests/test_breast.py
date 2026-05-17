import pandas as pd
df = pd.read_csv("../data/raw/breast_cancer_wisconsin.csv")
print("=== Surowe dane ===")
print("Kolumna diagnosis:", df['diagnosis'].unique())
print("Liczba M i B:", df['diagnosis'].value_counts())
print("Pierwsze 10 wierszy target:", df['diagnosis'].head(10))

from src.data_loader import load_dataset
X, y, config = load_dataset("breast_cancer")
print("=== Po binarizacji ===")
print("y unique:", y.unique())
print("y value_counts:", y.value_counts())

print("=== Kolumny z 'concave points' ===")
concave_cols = [col for col in df.columns if 'concave points' in col]
print(concave_cols)