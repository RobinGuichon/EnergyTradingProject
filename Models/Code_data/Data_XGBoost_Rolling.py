import pandas as pd
import numpy as np

path = "/Users/robinguichon/Desktop/ProjetEnerg/Datasets"

print("=== CLEAN V2 BUILD STARTED ===")

# === 1. Load raw dataset (merged) ===
df = pd.read_csv(f"{path}/dataset_merged_final.csv")

# === 2. Remove the fully NaN column ===
if "forecast wind offshore eday ahead" in df.columns:
    df = df.drop(columns=["forecast wind offshore eday ahead"])
    print("✔ Removed column: forecast wind offshore eday ahead")

# === 3. Convert time and sort ===
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# === 4. Fill small gaps ===
df = df.ffill().bfill()

# === 5. Rename target clearly ===
df["price_actual"] = df["price actual"]

# === 6. BUILD SAFE FEATURES (NO LEAKAGE) ===
# ------------------------------------------

# 6.1 PURE LAGS (OK)
df["lag_1"]   = df["price_actual"].shift(1)
df["lag_2"]   = df["price_actual"].shift(2)
df["lag_3"]   = df["price_actual"].shift(3)
df["lag_24"]  = df["price_actual"].shift(24)
df["lag_48"]  = df["price_actual"].shift(48)
df["lag_168"] = df["price_actual"].shift(168)

# 6.2 RETURNS (NO LEAK)
df["return_1h"] = df["lag_1"] - df["lag_2"]             # price[t-1] - price[t-2]
df["return_24h"] = df["lag_24"] - df["lag_48"]          # past diff only

# 6.3 ROLLING WINDOWS (NO LEAK)
df["roll_mean_24"]   = df["lag_1"].rolling(24).mean()
df["roll_std_24"]    = df["lag_1"].rolling(24).std()
df["roll_mean_168"]  = df["lag_1"].rolling(168).mean()
df["roll_std_168"]   = df["lag_1"].rolling(168).std()

# 6.4 VOLATILITY (NO LEAK)
df["vol_24"]  = df["return_1h"].rolling(24).std()
df["vol_168"] = df["return_1h"].rolling(168).std()

# 6.5 TREND (NO LEAK)
df["trend_24"]  = df["lag_1"] - df["lag_24"]
df["trend_168"] = df["lag_1"] - df["lag_168"]

# === 7. REMOVE any row with a NaN created by rolling windows ===
df = df.dropna().reset_index(drop=True)

# === 8. Build X and Y ===
Y = df[["price_actual"]]

cols_to_drop = [
    "price actual",      # old name
    "price_actual"       # remove target from X
]

X = df.drop(columns=cols_to_drop)

# === 9. SAVE ===
X.to_csv(f"{path}/X_clean_rolling_v2.csv", index=False)
Y.to_csv(f"{path}/Y_clean_v2.csv", index=False)

print("\n=== CLEAN V2 BUILD COMPLETE ===")
print("X_clean_rolling_v2 :", X.shape)
print("Y_clean_v2 :", Y.shape)
