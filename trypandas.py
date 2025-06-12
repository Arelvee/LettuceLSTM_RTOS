import pandas as pd

df = pd.read_csv("lettuce_wavelet_cleaned.csv")
print(df["growth_stage"].value_counts())