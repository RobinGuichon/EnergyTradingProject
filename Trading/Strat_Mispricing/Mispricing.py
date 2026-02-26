import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# ======================================================
# STYLE SETTINGS (APPLY TO ALL FIGURES)
# ======================================================
plt.rcParams["font.family"] = "Times New Roman"

yellow  = (231/255, 187/255, 65/255)
darkred = (103/255, 16/255, 9/255)

custom_cmap = LinearSegmentedColormap.from_list("custom_map", [darkred, yellow])

# ======================================================
# 0) PATHS
# ======================================================
model_path = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/Models/xgb_intraday_model.json"
save_path  = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/Trading_Strategies/Mispricing"
data_path  = "/Users/robinguichon/Desktop/ProjetEnerg/Data/Data_XGBoost_Rolling"

os.makedirs(save_path, exist_ok=True)

# ======================================================
# 1) LOAD MODEL
# ======================================================
model = xgb.Booster()
model.load_model(model_path)
print("✔ Model loaded.")

# ======================================================
# 2) LOAD FULL DATA
# ======================================================
X_full = pd.read_csv(f"{data_path}/X_clean_rolling_v2.csv")
Y_full = pd.read_csv(f"{data_path}/Y_clean_v2.csv")

X_full["time"] = pd.to_datetime(X_full["time"])

# ======================================================
# 3) EXTRACT LAST MONTH
# ======================================================
last_date = X_full["time"].max()
start_date = last_date - pd.Timedelta(days=30)

mask = X_full["time"] >= start_date
X_month = X_full.loc[mask].copy()
Y_month = Y_full.loc[mask].copy()

print("Last month shape:", X_month.shape)

X_month_features = X_month.drop(columns=["time"])

# ======================================================
# 4) PREDICTION
# ======================================================
dmonth = xgb.DMatrix(X_month_features)
y_pred_month = model.predict(dmonth)

price_real = Y_month.values.flatten()

price_t     = price_real[:-1]
price_tplus = price_real[1:]
pred_tplus  = y_pred_month[1:]
time_month  = X_month["time"].iloc[1:]

# ======================================================
# 5) MISPRICING STRATEGY
# ======================================================
spread = pred_tplus - price_tplus
positions = np.where(spread < 0, 1, -1)
pnl = positions * (price_tplus - price_t)
pnl_cum = np.cumsum(pnl)

# ======================================================
# 6) HOURLY PNL (YELLOW/GARNET)
# ======================================================
plt.figure(figsize=(12,6))
plt.bar(time_month, pnl, 
        color=[yellow if x > 0 else darkred for x in pnl], 
        width=0.06)
plt.title("Hourly PnL — Mispricing Strategy")
plt.xlabel("Time")
plt.ylabel("PnL (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_mispricing.png", dpi=300)

# ======================================================
# 7) CUMULATIVE PNL (YELLOW CURVE)
# ======================================================
plt.figure(figsize=(12,3))
plt.plot(time_month, pnl_cum, linewidth=2, color=yellow)
plt.title("Cumulative PnL — Mispricing Strategy")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_cum_mispricing.png", dpi=300)

# ======================================================
# 8) REAL vs PRED (STANDARD COLORS)
# ======================================================
plt.figure(figsize=(12,6))
plt.plot(time_month, price_tplus, label="Real Price", linewidth=2, color="black")
plt.plot(time_month, pred_tplus, label="Predicted Price", linewidth=2, color="blue")
plt.title("Real vs Predicted Price — Mispricing Strategy")
plt.xlabel("Time")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/real_vs_pred_mispricing.png", dpi=300)

# ======================================================
# 10) PNL DISTRIBUTION (YELLOW BINS)
# ======================================================
plt.figure(figsize=(12,6))
plt.hist(pnl, bins=50, color=yellow, edgecolor="black")
plt.title("PnL Distribution — Mispricing Strategy")
plt.xlabel("PnL (€)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_distribution_mispricing.png", dpi=300)

# ======================================================
# 11) HEATMAP (CUSTOM COLORMAP)
# ======================================================
hours = time_month.dt.hour.values
days  = time_month.dt.day.values

heatmap_matrix = np.full((24, 31), np.nan)
for h, d, p in zip(hours, days, pnl):
    heatmap_matrix[h, d-1] = 1 if p > 0 else 0

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_matrix, cmap=custom_cmap,
            xticklabels=range(1,32), yticklabels=range(0,24))
plt.title("Winrate Heatmap — Mispricing Strategy")
plt.xlabel("Day of Month")
plt.ylabel("Hour")
plt.tight_layout()
plt.savefig(f"{save_path}/heatmap_mispricing.png", dpi=300)

# ======================================================
# 12) CSV EXPORT
# ======================================================
df_trades = pd.DataFrame({
    "time": time_month.values,
    "price_t": price_t,
    "price_tplus": price_tplus,
    "pred_tplus": pred_tplus,
    "spread": spread,
    "position": positions,
    "pnl": pnl,
    "pnl_cum": pnl_cum
})

df_trades.to_csv(f"{save_path}/trades_mispricing.csv", index=False)
print(f"\n✔ CSV saved to : {save_path}/trades_mispricing.csv")
