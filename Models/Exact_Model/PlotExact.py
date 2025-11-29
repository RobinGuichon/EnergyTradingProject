import pandas as pd
import matplotlib.pyplot as plt

path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA1_LSTM"
path_out = "/Users/robinguichon/Desktop/ProjetEnerg/Plots"

# Load data
X = pd.read_csv(f"{path}/X_clean_LSTM.csv")
Y = pd.read_csv(f"{path}/Y_clean_LSTM.csv")

# Convert time to datetime
X["time"] = pd.to_datetime(X["time"])

# Rename target for clarity
Y = Y.rename(columns={"price actual": "price_intraday"})

# === Filter last N months ===
last_date = X["time"].max()
months_back = 36  # modify if needed

start_date = last_date - pd.DateOffset(months=months_back)
mask = X["time"] >= start_date

X_last = X.loc[mask]
Y_last = Y.loc[mask]

# === Plot Settings ===
plt.rcParams["font.family"] = "Times New Roman"

gold_color = (231/255, 187/255, 65/255)

plt.figure(figsize=(14, 6))
plt.plot(X_last["time"], Y_last["price_intraday"], 
         linewidth=0.4, color=gold_color)   # <-- Trait plus fin

plt.xlabel("Date", fontsize=13)
plt.ylabel("Intraday Price (€)", fontsize=13)
plt.title("Evolution of the Intraday Electricity Price (2015–2019)", fontsize=15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)

plt.tight_layout()

# === Save Figure ===
output_path = f"{path_out}/intraday_price_recent_period.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"✔ Figure saved to: {output_path}")

