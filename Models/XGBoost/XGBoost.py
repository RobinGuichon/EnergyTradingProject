import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Load Data ===
path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA1_XGBOOST"

X = pd.read_csv(f"{path}/X_final_clean.csv")
Y = pd.read_csv(f"{path}/Y_final.csv")

# Convert time column
X["time"] = pd.to_datetime(X["time"])

# Remove time column (non numerical)
X_features = X.drop(columns=["time"])

# === 2. Temporal Split ===
n = len(X_features)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

X_train = X_features.iloc[:train_size]
Y_train = Y.iloc[:train_size]

X_val = X_features.iloc[train_size: train_size + val_size]
Y_val = Y.iloc[train_size: train_size + val_size]

X_test = X_features.iloc[train_size + val_size:]
Y_test = Y.iloc[train_size + val_size:]

print("Shapes :")
print(X.shape)
print("Train :", X_train.shape, Y_train.shape)
print("Val   :", X_val.shape, Y_val.shape)
print("Test  :", X_test.shape, Y_test.shape)

# === 3. DMatrix Conversion ===
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_val, label=Y_val)
dtest = xgb.DMatrix(X_test, label=Y_test)

# === 4. BEST XGBoost Parameters (from Optuna 150 trials) ===
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.011079248646397246,
    "max_depth": 10,
    "min_child_weight": 18,
    "subsample": 0.5007868939044449,
    "colsample_bytree": 0.9985602201505985,
    "gamma": 0.6291347114905599,
    "lambda": 2.416950284412258,
    "alpha": 0.5183556110228869
}

watchlist = [(dtrain, "train"), (dval, "validation")]

# === 5. Train with Early Stopping ===
print("\nTraining XGBoost with BEST hyperparameters...")

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=watchlist,
    early_stopping_rounds=50,     # Robust for best model selection
    verbose_eval=50
)

print("\n✔ Training finished")
print("Best iteration:", model.best_iteration)

# === 6. Predictions ===
y_pred_train = model.predict(dtrain)
y_pred_val = model.predict(dval)
y_pred_test = model.predict(dtest)

# === 7. Metrics ===
def metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name} :")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

metrics(Y_train, y_pred_train, "Train")
metrics(Y_val, y_pred_val, "Validation")
metrics(Y_test, y_pred_test, "Test")

# === 8. Feature Importance (Gain) ===
importance = model.get_score(importance_type="gain")

importance_df = pd.DataFrame({
    "feature": list(importance.keys()),
    "importance": list(importance.values())
})

importance_df = importance_df.sort_values(by="importance", ascending=False)



# === 9. Real vs Predicted Plot (Test Set) — Times New Roman + Custom Colors + Save PNG ===
import matplotlib as mpl

# Global font
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

# Custom colors (convert to 0-1)
color_real = (103/255, 16/255, 9/255)
color_pred = (231/255, 187/255, 65/255)

time_test = X["time"].iloc[train_size + val_size:]

plt.figure(figsize=(12, 6))
plt.plot(time_test, Y_test.values, label="Real Price", linewidth=0.8, color=color_real)
plt.plot(time_test, y_pred_test, label="Predicted Price", linewidth=0.8, color=color_pred)

plt.title("Real vs Predicted Electricity Price — XGBoost_Lags (Test Set)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()


save_path = "/Users/robinguichon/Desktop/ProjetEnergPlots/xgb_lags_test.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✔ Test-set figure saved at: {save_path}")

# === 10. Last Month Zoom Plot — Times New Roman + Custom Colors + Save PNG ===

# 30 days = 720 hours
last_month_hours = 24 * 7

X_last_month = time_test.iloc[-last_month_hours:]
Y_last_month = Y_test.values[-last_month_hours:]
pred_last_month = y_pred_test[-last_month_hours:]

plt.figure(figsize=(12, 6))
plt.plot(X_last_month, Y_last_month, label="Real Price", linewidth=1.2, color=color_real)
plt.plot(X_last_month, pred_last_month, label="Predicted Price", linewidth=1.2, color=color_pred)

plt.title("Real vs Predicted Electricity Price — XGBoost_Lags (Last Week)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path_last = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/xgb_lags_last_month.png"
plt.savefig(save_path_last, dpi=300, bbox_inches="tight")
plt.show()

print(f"✔ Last-month figure saved at: {save_path_last}")



# === 10. Additional Metrics ===

# Avoid division by zero for MAPE
mask = Y_test.values.flatten() != 0
y_true_nozero = Y_test.values.flatten()[mask]
y_pred_nozero = y_pred_test[mask]

# === MAPE ===
mape = np.mean(np.abs((y_true_nozero - y_pred_nozero) / y_true_nozero)) * 100

# === R² ===
r2 = r2_score(Y_test, y_pred_test)

# === MASE ===
naive_forecast = Y_test.shift(1).iloc[1:].values.flatten()
true_aligned = Y_test.iloc[1:].values.flatten()
mae_naive = np.mean(np.abs(true_aligned - naive_forecast))
mae_model = mean_absolute_error(true_aligned, y_pred_test[1:])
mase = mae_model / mae_naive

# === Directional Accuracy ===
true_diff = np.sign(Y_test.diff().iloc[1:].values.flatten())
pred_diff = np.sign(np.diff(y_pred_test))
directional_accuracy = np.mean(true_diff == pred_diff) * 100

print("\n=== Additional Metrics ===")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print(f"MASE : {mase:.4f}")
print(f"Directional Accuracy : {directional_accuracy:.2f}%")
