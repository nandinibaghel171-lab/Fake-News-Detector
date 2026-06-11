import pandas as pd

# Data load karo
df = pd.read_csv('data/fake.csv')

# Basic info dekho
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())