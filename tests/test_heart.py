import pandas as pd
df_heart = pd.read_csv("../data/raw/heart_disease.csv")
print("=== Surowe num ===")
print("num unique:", df_heart['num'].unique())
print("num value_counts:", df_heart['num'].value_counts())

from src.data_loader import load_dataset
x_heart, y_heart, _ = load_dataset("heart")
print("=== Po binarizacji heart ===")
print("y_heart unique:", y_heart.unique())
print("y_heart value_counts:", y_heart.value_counts())

print("Heart: num > 0 =", y_heart.sum(), f"({y_heart.sum()/len(y_heart)*100:.1f}%)")
