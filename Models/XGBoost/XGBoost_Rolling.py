import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Load Data ===
path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA2_XGBOOST"

X = pd.read_csv(f"{path}/X_clean_rolling_v2.csv")
Y = pd.read_csv(f"{path}/Y_clean_v2.csv")

# Convert time back to datetime
X["time"] = pd.to_datetime(X["time"])

# Remove time for modeling
X_features = X.drop(columns=["time"])

# === 2. Temporal Split ===
n = len(X_features)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

X_train = X_features.iloc[:train_size]
Y_train = Y.iloc[:train_size]

X_val = X_features.iloc[train_size : train_size + val_size]
Y_val = Y.iloc[train_size : train_size + val_size]

X_test = X_features.iloc[train_size + val_size :]
Y_test = Y.iloc[train_size + val_size :]

print("Shapes:")
print("Train:", X_train.shape, Y_train.shape)
print("Val:  ", X_val.shape, Y_val.shape)
print("Test: ", X_test.shape, Y_test.shape)

# === 3. DMatrix ===
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval   = xgb.DMatrix(X_val, label=Y_val)
dtest  = xgb.DMatrix(X_test, label=Y_test)

# === 4. Hyperparameters (Optuna Best) ===
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.0052856998613360455,
    "max_depth": 9,
    "min_child_weight": 11,
    "subsample": 0.5370858622948435,
    "colsample_bytree": 0.9535031113477612,
    "gamma": 2.2331918753326874,
    "lambda": 0.23912890666433612,
    "alpha": 1.518891405996986,
}

watchlist = [(dtrain, "train"), (dval, "val")]

# === 5. Train ===
print("\nTraining XGBoost...")

model = xgb.train(
    params=params,
    dtrain=dtrain,
    evals=watchlist,
    num_boost_round=3000,
    early_stopping_rounds=80,
    verbose_eval=50
)

print("\n✔ Training complete")
print("Best iteration:", model.best_iteration)

# === 6. Predictions ===
y_pred_train = model.predict(dtrain)
y_pred_val   = model.predict(dval)
y_pred_test  = model.predict(dtest)

# === 7. Metrics ===
def metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

metrics(Y_train, y_pred_train, "Train")
metrics(Y_val,   y_pred_val,   "Validation")
metrics(Y_test,  y_pred_test,  "Test")

# === 8. Additional Metrics ===

mask = Y_test.values.flatten() != 0
true_no0 = Y_test.values.flatten()[mask]
pred_no0 = y_pred_test[mask]

# MAPE
mape = np.mean(np.abs((true_no0 - pred_no0) / true_no0)) * 100

# R²
r2 = r2_score(Y_test, y_pred_test)

# MASE
naive = Y_test.shift(1).iloc[1:].values.flatten()
true_aligned = Y_test.iloc[1:].values.flatten()
mae_naive = np.mean(np.abs(true_aligned - naive))
mae_model = mean_absolute_error(true_aligned, y_pred_test[1:])
mase = mae_model / mae_naive

# Directional Accuracy
true_diff = np.sign(Y_test.diff().iloc[1:].values.flatten())
pred_diff = np.sign(np.diff(y_pred_test))
DA = np.mean(true_diff == pred_diff) * 100

print("\n=== Additional Metrics ===")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print(f"MASE : {mase:.4f}")
print(f"Directional Accuracy : {DA:.2f}%")

# === 9. Plot Test Predictions + Save Figure ===
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set Times New Roman for the entire plot
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

time_test = X["time"].iloc[train_size + val_size :]

plt.figure(figsize=(12, 6))

# Convert RGB 0–255 → 0–1
color_real = (103/255, 16/255, 9/255)
color_pred = (231/255, 187/255, 65/255)

plt.plot(time_test, Y_test.values,
         label="Real Price",
         linewidth=0.6,
         color=color_real)

plt.plot(time_test, y_pred_test,
         label="Predicted Price",
         linewidth=0.6,
         color=color_pred)

plt.title("Real vs Predicted Price — XGBoost_Rolling (Test Set)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save figure as PNG ===
save_path = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/xgbroll_test_predictions.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"\n✔ Figure saved successfully at:\n{save_path}")



# === 10. Plot Last-Month Predictions + Save Figure ===

# Extract last 30 days (720 hours)
# Assumes hourly data — adjust 24*30 if different resolution
last_month_hours = 24 * 7  
X_last_month = time_test.iloc[-last_month_hours:]
Y_last_month = Y_test.values[-last_month_hours:]
y_pred_last_month = y_pred_test[-last_month_hours:]

plt.figure(figsize=(12, 6))

# Times New Roman already set by rcParams above
plt.plot(X_last_month, Y_last_month,
         label="Real Price",
         linewidth=1.2,
         color=color_real)

plt.plot(X_last_month, y_pred_last_month,
         label="Predicted Price",
         linewidth=1.2,
         color=color_pred)

plt.title("Real vs Predicted Price — XGBoost_Rolling (Last Week)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
save_path_last = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/xgbroll_last_month.png"
plt.savefig(save_path_last, dpi=300, bbox_inches="tight")

plt.show()

print(f"\n✔ Last-month figure saved successfully at:\n{save_path_last}")
