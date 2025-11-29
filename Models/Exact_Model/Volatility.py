import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Load dataset
# ============================
df = pd.read_csv("/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA0/dataset_merged_final.csv")

# Detect datetime column
datetime_col = None
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        datetime_col = col
        break

if datetime_col is None:
    datetime_col = df.columns[0]

df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.sort_values(datetime_col)

# Identify price column
price_candidates = [c for c in df.columns if "price" in c.lower()]
price_col = price_candidates[0] if price_candidates else df.columns[1]

# ============================
# Compute returns
# ============================
df["diff"] = df[price_col].diff()
df["pct"] = df[price_col].pct_change()
df["logret"] = np.log(df[price_col] / df[price_col].shift(1))
df["absret"] = df["diff"].abs()
df["hour"] = df[datetime_col].dt.hour

# ============================
# Compute volatilities
# ============================
std_diff = df.groupby("hour")["diff"].std()
std_pct = df.groupby("hour")["pct"].std()
std_log = df.groupby("hour")["logret"].std()
mean_abs = df.groupby("hour")["absret"].mean()

# ============================
# Custom colors (your colors)
# ============================
colors = {
    "std_diff": (103/255, 16/255, 9/255),      # dark red
    "std_pct":  (154/255, 67/255, 28/255),     # brownish red
    "std_log":  (184/255, 114/255, 42/255),    # orange-brown
    "mean_abs": (231/255, 187/255, 65/255)     # gold
}

# ============================
# Plot
# ============================
plt.rcParams["font.family"] = "Times New Roman"

plt.figure(figsize=(12, 6))

plt.plot(std_diff.index, std_diff.values, color=colors["std_diff"], linewidth=2.5,
         label="Std of Price Differences")

plt.plot(std_pct.index, std_pct.values, color=colors["std_pct"], linewidth=2.3,
         label="Std of Percentage Returns")

plt.plot(std_log.index, std_log.values, color=colors["mean_abs"], linewidth=2.3,
         label="Std of Log Returns")

plt.plot(mean_abs.index, mean_abs.values, color=colors["std_log"], linewidth=2.3,
         label="Mean Absolute Differences")

plt.xlabel("Hour of Day", fontsize=14)
plt.ylabel("Volatility Measure", fontsize=14)
plt.title("Intraday Volatility Measures (Comparison)", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=13)
plt.tight_layout()

# ============================
# Save + Trigger download
# ============================
output_path = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/intraday_volatility_models.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
output_path
