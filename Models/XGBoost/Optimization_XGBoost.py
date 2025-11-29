import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# === 1. Load Data ===
path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA1_XGBOOST"

X = pd.read_csv(f"{path}/X_final_clean.csv")
Y = pd.read_csv(f"{path}/Y_final.csv")

X["time"] = pd.to_datetime(X["time"])
X_features = X.drop(columns=["time"])

# === 2. Temporal Split ===
n = len(X_features)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

X_train = X_features.iloc[:train_size]
Y_train = Y.iloc[:train_size]

X_val = X_features.iloc[train_size:train_size + val_size]
Y_val = Y.iloc[train_size:train_size + val_size]

X_test = X_features.iloc[train_size + val_size:]
Y_test = Y.iloc[train_size + val_size:]

print("Shapes :")
print("Train :", X_train.shape, Y_train.shape)
print("Val   :", X_val.shape, Y_val.shape)
print("Test  :", X_test.shape, Y_test.shape)

# === 3. Create DMatrix ===
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_val, label=Y_val)
dtest = xgb.DMatrix(X_test, label=Y_test)

# === 4. Baseline Model (optional) ===
params_baseline = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

print("\nTraining baseline model...")

baseline = xgb.train(
    params=params_baseline,
    dtrain=dtrain,
    evals=[(dtrain, "train"), (dval, "validation")],
    num_boost_round=2000,
    early_stopping_rounds=50,
    verbose_eval=50
)

print("\nBaseline RMSE validation:", baseline.best_score)

# === 5. OPTUNA OPTIMIZATION ===
def objective(trial):
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": trial.suggest_float("eta", 0.01, 0.30),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "lambda": trial.suggest_float("lambda", 0.0, 5.0),
        "alpha": trial.suggest_float("alpha", 0.0, 5.0)
    }

    model = xgb.train(
        params=param,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "validation")],
        num_boost_round=2000,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    return model.best_score

print("\n🔥 Running Optuna (150 trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150)

print("\n=== Best Hyperparameters ===")
print(study.best_params)
print("\nBest RMSE validation:", study.best_value)

# === 6. Train BEST MODEL ===
best_params = study.best_params
best_params["objective"] = "reg:squarederror"
best_params["eval_metric"] = "rmse"

print("\nTraining BEST MODEL...")

best_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    evals=[(dtrain, "train"), (dval, "validation")],
    num_boost_round=2000,
    early_stopping_rounds=50,
    verbose_eval=50
)

print("\nBest iteration:", best_model.best_iteration)

# === 7. Predictions ===
y_pred_train = best_model.predict(dtrain)
y_pred_val = best_model.predict(dval)
y_pred_test = best_model.predict(dtest)

# === 8. Metrics ===
def metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

metrics(Y_train, y_pred_train, "Train")
metrics(Y_val, y_pred_val, "Validation")
metrics(Y_test, y_pred_test, "Test")

# === Additional Metrics ===

mask = Y_test.values.flatten() != 0
y_true_nozero = Y_test.values.flatten()[mask]
y_pred_nozero = y_pred_test[mask]

mape = np.mean(np.abs((y_true_nozero - y_pred_nozero) / y_true_nozero)) * 100
r2 = r2_score(Y_test, y_pred_test)

naive = Y_test.shift(1).iloc[1:].values.flatten()
aligned = Y_test.iloc[1:].values.flatten()
mae_naive = np.mean(np.abs(aligned - naive))
mae_model = mean_absolute_error(aligned, y_pred_test[1:])
mase = mae_model / mae_naive

true_diff = np.sign(Y_test.diff().iloc[1:].values.flatten())
pred_diff = np.sign(np.diff(y_pred_test))
DA = np.mean(true_diff == pred_diff) * 100

print("\n=== Additional Metrics ===")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print(f"MASE : {mase:.4f}")
print(f"Directional Accuracy : {DA:.2f}%")

# === 9. Plot Real vs Predicted (Test) ===
time_test = X["time"].iloc[train_size + val_size:]

plt.figure(figsize=(14,6))
plt.plot(time_test, Y_test.values, label="Real Price", color="blue", linewidth=1)
plt.plot(time_test, y_pred_test, label="Predicted Price", color="red", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Real vs Predicted Electricity Price (TEST)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 10. Feature Importance ===
importance = best_model.get_score(importance_type="gain")
importance_df = pd.DataFrame({
    "feature": list(importance.keys()),
    "importance": list(importance.values())
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 7))
plt.barh(importance_df["feature"].head(25), importance_df["importance"].head(25))
plt.gca().invert_yaxis()
plt.title("Top 25 Feature Importances (Gain)")
plt.tight_layout()
plt.show()
