import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# GLOBAL FIGURE STYLE (Times, Yellow, Dark Red)
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"

yellow  = (231/255, 187/255, 65/255)
darkred = (103/255, 16/255, 9/255)

custom_cmap = LinearSegmentedColormap.from_list("custom_map", [darkred, yellow])

# =========================================================
# 1) LOAD TRAINED MODEL
# =========================================================
model_path = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/Modèles/xgb_intraday_model.json"
model = xgb.Booster()
model.load_model(model_path)

print(f"✔ Loaded model: {model_path}")

# =========================================================
# 2) LOAD DATA
# =========================================================
path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA2_XGBOOST"

X_full = pd.read_csv(f"{path}/X_clean_rolling_v2.csv")
Y_full = pd.read_csv(f"{path}/Y_clean_v2.csv")

X_full["time"] = pd.to_datetime(X_full["time"])

# =========================================================
# 3) LAST MONTH EXTRACTION
# =========================================================
last_date = X_full["time"].max()
start_last_month = last_date - pd.Timedelta(days=30)

mask = X_full["time"] >= start_last_month

X_month = X_full.loc[mask].copy()
Y_month = Y_full.loc[mask].copy()

print("Last month data shape:", X_month.shape)

# Prepare features
X_month_features = X_month.drop(columns=["time"])
dmonth = xgb.DMatrix(X_month_features)

# =========================================================
# 4) PREDICTION
# =========================================================
y_pred_month = model.predict(dmonth)
price_real = Y_month.values.flatten()

price_t     = price_real[:-1]
price_tplus = price_real[1:]
pred_tplus  = y_pred_month[1:]
time_month  = X_month["time"].iloc[1:]

# =========================================================
# 5) DIRECTIONAL STRATEGY
# =========================================================
positions = np.where(pred_tplus > price_t, 1, -1)

# =========================================================
# 6) PNL COMPUTATION
# =========================================================
pnl = positions * (price_tplus - price_t)
pnl_cum = np.cumsum(pnl)

initial_capital = 0
capital_curve = initial_capital + pnl_cum

# =========================================================
# SAVE PATH
# =========================================================
save_path = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/Trade1"
os.makedirs(save_path, exist_ok=True)

# =========================================================
# 7) HOURLY PNL — Yellow / Dark Red
# =========================================================
plt.figure(figsize=(12,6))
plt.bar(time_month, pnl, 
        color=[yellow if x > 0 else darkred for x in pnl],
        width=0.06)
plt.title("Hourly PnL — Directional Strategy")
plt.xlabel("Time")
plt.ylabel("PnL (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_hourly.png", dpi=300)

# =========================================================
# 8) CUMULATIVE PNL — Yellow Curve
# =========================================================
plt.figure(figsize=(12,3))
plt.plot(time_month, pnl_cum, linewidth=2, color=yellow)
plt.title("Cumulative PnL — Directional Strategy")
plt.xlabel("Time")
plt.ylabel("Cumulative PnL (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_cum.png", dpi=300)

# =========================================================
# 9) REAL vs PREDICTED PRICE
# =========================================================
plt.figure(figsize=(12,6))
plt.plot(time_month, price_tplus, label="Real Price", linewidth=2, color="black")
plt.plot(time_month, pred_tplus, label="Predicted Price", linewidth=2, color="blue")
plt.title("Real vs Predicted Price — Last Month")
plt.xlabel("Time")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/real_vs_pred.png", dpi=300)

# =========================================================
# 10) PRINT RESULTS
# =========================================================
capital_initial = 0
capital_final = pnl_cum[-1]

print("\n=== Backtest Results ===")
print(f"Initial Capital : {capital_initial} €")
print(f"Final Capital   : {capital_final:.2f} €")
print(f"Total Profit    : {capital_final:.2f} €")
print(f"Number of Trades: {len(pnl)}")
print(f"Winrate         : {np.mean(pnl > 0)*100:.2f}%")

# Sharpe Ratio
mean_pnl = np.mean(pnl)
std_pnl  = np.std(pnl)
sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(24 * 365) if std_pnl != 0 else 0
print(f"Sharpe Ratio    : {sharpe_ratio:.4f}")

# Max Drawdown
running_max = np.maximum.accumulate(pnl_cum)
drawdown = pnl_cum - running_max
max_drawdown = drawdown.min()
print(f"Max Drawdown    : {max_drawdown:.2f} €")

# =========================================================
# 11) PNL DISTRIBUTION — Yellow bars
# =========================================================
plt.figure(figsize=(12,6))
plt.hist(pnl, bins=50, color=yellow, edgecolor="black")
plt.title("PnL Distribution — Directional Strategy")
plt.xlabel("PnL (€)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_distribution.png", dpi=300)

# =========================================================
# 12) HEATMAP WINRATE — Using Custom Yellow-Red Map
# =========================================================
hours = time_month.dt.hour.values
days  = time_month.dt.day.values

heatmap_matrix = np.full((24, 31), np.nan)
for h, d, p in zip(hours, days, pnl):
    heatmap_matrix[h, d-1] = 1 if p > 0 else 0

plt.figure(figsize=(12,6))
sns.heatmap(
    heatmap_matrix,
    cmap=custom_cmap,
    xticklabels=range(1,32),
    yticklabels=range(0,24)
)
plt.title("Winrate Heatmap — Directional Strategy")
plt.xlabel("Day of Month")
plt.ylabel("Hour")
plt.tight_layout()
plt.savefig(f"{save_path}/heatmap_winrate.png", dpi=300)

# =========================================================
# 13) EXPORT TRADES CSV
# =========================================================
df_trades = pd.DataFrame({
    "time": time_month.values,
    "price_t": price_t,
    "price_tplus": price_tplus,
    "pred_tplus": pred_tplus,
    "position": positions,
    "pnl": pnl,
    "pnl_cum": pnl_cum
})

output_csv = f"{save_path}/trades_last_month.csv"
df_trades.to_csv(output_csv, index=False)

print(f"\n✔ Trades CSV saved: {output_csv}")
