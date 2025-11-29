import pandas as pd

# === 1. Load full dataset ===
path = "/Users/robinguichon/Desktop/ProjetEnerg/Datasets"
df = pd.read_csv(f"{path}/dataset_merged_final.csv")

print("Taille initiale :", df.shape)

# === 2. Clean ===
df = df.ffill().bfill()

# === 3. Convert time ===
df["time"] = pd.to_datetime(df["time"])

# === 4. Remove first 168 rows to allow lag creation ===
df = df.iloc[168:].reset_index(drop=True)
print("Après suppression des 168 premières lignes :", df.shape)

# === 5. Create lags ===
lags = [1, 2, 3, 6, 12, 24, 48, 168]

for lag in lags:
    df[f"price_actual_lag_{lag}"] = df["price actual"].shift(lag)

# Remove only NEW lag NaN rows
df = df.dropna(subset=[f"price_actual_lag_{lag}" for lag in lags])
print("Après création des lags :", df.shape)

# === 6. Build X and Y ===
Y = df["price actual"]
X = df.drop(columns=["price actual"])

# === 7. Save ===
save = "/Users/robinguichon/Desktop/ProjetEnerg"
X.to_csv(f"{save}/X_final.csv", index=False)
Y.to_csv(f"{save}/Y_final.csv", index=False)

print("✔ X_final et Y_final créés avec succès.")
