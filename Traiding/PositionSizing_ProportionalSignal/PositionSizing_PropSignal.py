import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# ======================================================
# GLOBAL STYLE : Times New Roman + Custom Colors
# ======================================================
plt.rcParams["font.family"] = "Times New Roman"

yellow  = (231/255, 187/255, 65/255)
darkred = (103/255, 16/255, 9/255)

custom_cmap = LinearSegmentedColormap.from_list("custom_map", [darkred, yellow])

# ======================================================
# PATHS
# ======================================================
model_path = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/Modèles/xgb_intraday_model.json"
save_path  = "/Users/robinguichon/Desktop/ProjetEnerg/Traiding/PositionSizing_ProportionnelSignal"
data_path  = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA2_XGBOOST"

os.makedirs(save_path, exist_ok=True)

# ======================================================
# LOAD MODEL
# ======================================================
model = xgb.Booster()
model.load_model(model_path)
print("✔ Model loaded.")

# ======================================================
# LOAD DATA
# ======================================================
X_full = pd.read_csv(f"{data_path}/X_clean_rolling_v2.csv")
Y_full = pd.read_csv(f"{data_path}/Y_clean_v2.csv")
X_full["time"] = pd.to_datetime(X_full["time"])

# ======================================================
# LAST MONTH EXTRACTION
# ======================================================
last_date = X_full["time"].max()
start_date = last_date - pd.Timedelta(days=30)

mask = X_full["time"] >= start_date
X_month = X_full.loc[mask].copy()
Y_month = Y_full.loc[mask].copy()

print("Last month shape:", X_month.shape)

X_features = X_month.drop(columns=["time"])
dmonth = xgb.DMatrix(X_features)
y_pred_month = model.predict(dmonth)

price_real = Y_month.values.flatten()

# Align for t and t+1
price_t     = price_real[:-1]
price_tplus = price_real[1:]
pred_tplus  = y_pred_month[1:]
time_month  = X_month["time"].iloc[1:]

# ======================================================
# TRADE 4 — HYBRID POSITION SIZING (CAPPED)
# ======================================================
capital0 = 1_000_000
capital = capital0

signal = pred_tplus - price_t
mispricing = pred_tplus - price_tplus

intensity = np.abs(signal) * np.abs(mispricing)
scaling = 20
raw_alloc = intensity / scaling
allocation = np.minimum(raw_alloc, 0.3)

positions = np.where(signal > 0, 1, -1)
MAX_POSITION_VALUE = 10000

capital_list = [capital]
pnl_list = []

for i in range(len(price_t)):
    theo_expo = allocation[i] * capital
    exposure = min(theo_expo, MAX_POSITION_VALUE)

    price_change_percent = (price_tplus[i] - price_t[i]) / price_t[i]
    pnl = positions[i] * exposure * price_change_percent

    capital += pnl
    pnl_list.append(pnl)
    capital_list.append(capital)

capital_list = np.array(capital_list)
pnl_list = np.array(pnl_list)
returns = pnl_list / capital_list[:-1]

# ======================================================
# STATISTICS
# ======================================================
profit_total = capital - capital0
winrate = np.mean(pnl_list > 0) * 100

if np.std(returns) != 0:
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
else:
    sharpe = 0

running_max = np.maximum.accumulate(capital_list)
drawdown = capital_list - running_max
max_dd = drawdown.min()

print("\n=== TRADE 4 — HYBRID POSITION SIZING (CAPPED) ===")
print(f"Initial Capital : {capital0:,.2f} €")
print(f"Final Capital   : {capital:,.2f} €")
print(f"Total Profit    : {profit_total:,.2f} €")
print(f"Winrate         : {winrate:.2f}%")
print(f"Sharpe Ratio    : {sharpe:.4f}")
print(f"Max Drawdown    : {max_dd:,.2f} €")
print(f"Trades          : {len(pnl_list)}")

# ======================================================
# FIGURES (UNIFIED STYLE)
# ======================================================

# ---- FIGURE 1 : HOURLY PNL ----
plt.figure(figsize=(12,6))
plt.bar(time_month, pnl_list,
        color=[yellow if x>0 else darkred for x in pnl_list],
        width=0.06)
plt.title("Hourly PnL — Hybrid Position Sizing (Capped)")
plt.xlabel("Time")
plt.ylabel("PnL (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_hourly_trade4.png", dpi=300)

# ---- FIGURE 2 : CAPITAL CURVE ----
plt.figure(figsize=(12,3))
plt.plot(time_month, capital_list[1:], color=yellow, linewidth=2)
plt.title("Capital Curve — Hybrid Position Sizing (Capped)")
plt.xlabel("Time")
plt.ylabel("Capital (€)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/capital_trade4.png", dpi=300)

# ---- FIGURE 3 : REAL VS PRED ----
plt.figure(figsize=(12,6))
plt.plot(time_month, price_tplus, linewidth=2, color="black", label="Real Price")
plt.plot(time_month, pred_tplus, linewidth=2, color="blue", label="Predicted Price")
plt.title("Real vs Predicted Price — Hybrid Position Sizing")
plt.xlabel("Time")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/real_vs_pred_trade4.png", dpi=300)

# ---- FIGURE 4 : PNL DISTRIBUTION ----
plt.figure(figsize=(12,6))
plt.hist(pnl_list, bins=50, color=yellow, edgecolor="black")
plt.title("PnL Distribution — Hybrid Position Sizing")
plt.xlabel("PnL (€)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_path}/pnl_distribution_trade4.png", dpi=300)

# ---- FIGURE 5 : HEATMAP ----
hours = time_month.dt.hour.values
days  = time_month.dt.day.values

heatmap_matrix = np.full((24, 31), np.nan)
for h, d, p in zip(hours, days, pnl_list):
    heatmap_matrix[h, d-1] = 1 if p > 0 else 0

plt.figure(figsize=(12,6))
sns.heatmap(
    heatmap_matrix,
    cmap=custom_cmap,
    xticklabels=range(1,32),
    yticklabels=range(0,24)
)
plt.title("Winrate Heatmap — Hybrid Position Sizing (Capped)")
plt.xlabel("Day of Month")
plt.ylabel("Hour")
plt.tight_layout()
plt.savefig(f"{save_path}/heatmap_trade4.png", dpi=300)

# ======================================================
# EXPORT CSV
# ======================================================
df_export = pd.DataFrame({
    "time": time_month.values,
    "price_t": price_t,
    "price_tplus": price_tplus,
    "pred_tplus": pred_tplus,
    "signal": signal,
    "mispricing": mispricing,
    "allocation": allocation,
    "exposure": np.minimum(allocation * capital0, MAX_POSITION_VALUE),
    "position": positions,
    "pnl": pnl_list,
    "capital": capital_list[1:]
})

df_export.to_csv(f"{save_path}/trades_trade4.csv", index=False)
print(f"\n✔ CSV saved : {save_path}/trades_trade4.csv")
