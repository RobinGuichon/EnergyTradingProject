import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

# ======================================================
# FONT SETTINGS : Times New Roman everywhere
# ======================================================
plt.rcParams["font.family"] = "Times New Roman"

# Yellow & dark red (your colors)
yellow = (231/255, 187/255, 65/255)
darkred = (103/255, 16/255, 9/255)

# Custom colormap for heatmap
custom_cmap = LinearSegmentedColormap.from_list("custom_map", [darkred, yellow])

# ======================================================
# 0) PATHS
# ======================================================
model_path = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/Modèles/xgb_intraday_model.json"
save_path  = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/Trade2"
data_path  = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA2_XGBOOST"

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

# Remove time column for prediction
X_month_features = X_month.drop(columns=["time"])

# ======================================================
# 4) PREDICTION
# ======================================================
dmonth = xgb.DMatrix(X_month_features)
y_pred_month = model.predict(dmonth)

price_real = Y_month.values.flatten()

# Align t and t+1
price_t     = price_real[:-1]
price_tplus = price_real[1:]
pred_tplus  = y_pred_month[1:]
time_month  = X_month["time"].iloc[1:]

# ======================================================
# 5) GRID SEARCH FOR BEST THRESHOLD
# ======================================================
thresholds = [0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5]

results = []

for th in thresholds:
    signal = pred_tplus - price_t
    positions = np.where(signal > th, 1,
                  np.where(signal < -th, -1, 0))
    
    pnl = positions * (price_tplus - price_t)
    pnl_cum = np.cumsum(pnl)
    
    total_pnl = pnl_cum[-1]
    winrate = np.mean(pnl > 0) * 100
    num_trades = np.sum(positions != 0)
    
    sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(24 * 365) if np.std(pnl) != 0 else 0
    
    running_max = np.maximum.accumulate(pnl_cum)
    drawdown = pnl_cum - running_max
    max_dd = drawdown.min()
    
    results.append([th, total_pnl, winrate, num_trades, sharpe, max_dd])

df_results = pd.DataFrame(results, columns=[
    "threshold", "total_pnl", "winrate", "num_trades", "sharpe", "max_drawdown"
])

print("\n=== GRID SEARCH RESULTS ===")
print(df_results)

df_results.to_csv(f"{save_path}/grid_search_thresholds.csv", index=False)

best_threshold = df_results.iloc[df_results["total_pnl"].idxmax()]["threshold"]
print(f"\n✔ Best threshold = {best_threshold}")

# ======================================================
# 6) BACKTEST WITH BEST THRESHOLD
# ======================================================
signal = pred_tplus - price_t

positions = np.where(signal > best_threshold, 1,
              np.where(signal < -best_threshold, -1, 0))

pnl = positions * (price_tplus - price_t)
pnl_cum = np.cumsum(pnl)

# ======================================================
# 7) PLOT : HOURLY PNL
# ======================================================
plt.figure(figsize=(12,6))
plt.bar(time_month, pnl, 
        color=[yellow if x > 0 else darkred for x in pnl], 
        width=0.06)
plt.title(f"Hourly PnL — Threshold = {best_threshold}")
plt.xlabel("Time")
plt.ylabel("PnL")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_hourly_threshold.png", dpi=300)

# ======================================================
# 8) PLOT : CUMULATIVE PNL (YELLOW CURVE)
# ======================================================
plt.figure(figsize=(12,3))
plt.plot(time_month, pnl_cum, linewidth=2, color=yellow)
plt.title(f"Cumulative PnL — Threshold = {best_threshold}")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_cum_threshold.png", dpi=300)

# ======================================================
# 9) REAL vs PRED
# ======================================================
plt.figure(figsize=(12,6))
plt.plot(time_month, price_tplus, label="Real Price", linewidth=1.5)
plt.plot(time_month, pred_tplus, label="Predicted Price", linewidth=1.5)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/real_vs_pred_threshold.png", dpi=300)

# ======================================================
# 11) PNL DISTRIBUTION (YELLOW BARS)
# ======================================================
plt.figure(figsize=(12,6))
plt.hist(pnl, bins=50, color=yellow, edgecolor="black")
plt.title("PnL Distribution — Threshold")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_distribution_threshold.png", dpi=300)

# ======================================================
# 12) HEATMAP (CUSTOM COLOR 1 & 2)
# ======================================================
hours = time_month.dt.hour.values
days  = time_month.dt.day.values

heatmap_matrix = np.full((24, 31), np.nan)

for h, d, p in zip(hours, days, pnl):
    heatmap_matrix[h, d-1] = 1 if p > 0 else 0

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_matrix, cmap=custom_cmap, 
            xticklabels=range(1,32), yticklabels=range(0,24))
plt.title("Heatmap Winrate — Threshold")
plt.xlabel("Day")
plt.ylabel("Hour")
plt.tight_layout()
plt.savefig(f"{save_path}/heatmap_threshold.png", dpi=300)
