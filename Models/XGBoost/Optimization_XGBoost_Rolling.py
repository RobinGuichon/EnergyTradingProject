import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# === 1. LOAD DATA ===

path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA2_XGBOOST"

X = pd.read_csv(f"{path}/X_clean_rolling_v2.csv")
Y = pd.read_csv(f"{path}/Y_clean_v2.csv")

# Convert time
X["time"] = pd.to_datetime(X["time"])

# Remove time column
X_features = X.drop(columns=["time"])

# === 2. TEMPORAL SPLIT ===

n = len(X_features)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

X_train = X_features.iloc[:train_size]
Y_train = Y.iloc[:train_size]

X_val = X_features.iloc[train_size:train_size + val_size]
Y_val = Y.iloc[train_size:train_size + val_size]

X_test = X_features.iloc[train_size + val_size:]
Y_test = Y.iloc[train_size + val_size:]

print("Shapes:")
print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape, Y_val.shape)
print("Test :", X_test.shape, Y_test.shape)

# === Convert to DMatrix ===
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_val, label=Y_val)
dtest = xgb.DMatrix(X_test, label=Y_test)

watchlist = [(dtrain, "train"), (dval, "validation")]


# ============================================================
#                 OPTUNA SEARCH SPACE (300 TRIALS)
# ============================================================

def objective(trial):

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",

        # Learning rate
        "eta": trial.suggest_float("eta", 0.005, 0.15),

        # Tree complexity
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 25),

        # Sampling (rolling data often benefits)
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # Regularization
        "lambda": trial.suggest_float("lambda", 0.1, 5.0),
        "alpha": trial.suggest_float("alpha", 0.0, 5.0),

        # Smoothing splits
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    return rmse


# === Run Optuna study ===

print("\n🔥 Running Optuna (300 trials)...")

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300, show_progress_bar=True)

print("\n=== BEST PARAMETERS ===")
print(study.best_params)
print("Best RMSE:", study.best_value)

best_params = study.best_params
best_params["objective"] = "reg:squarederror"
best_params["eval_metric"] = "rmse"

# ============================================================
#              TRAIN BEST MODEL (FROM OPTUNA)
# ============================================================

print("\nTraining BEST model...")

best_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=3000,
    evals=watchlist,
    early_stopping_rounds=80,
    verbose_eval=50
)

print("\nBest iteration:", best_model.best_iteration)

# === Predictions ===
pred_train = best_model.predict(dtrain)
pred_val = best_model.predict(dval)
pred_test = best_model.predict(dtest)

# === Metrics utility ===
def metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")


metrics(Y_train, pred_train, "Train")
metrics(Y_val, pred_val, "Validation")
metrics(Y_test, pred_test, "Test")

# === Additional Metrics ===
from sklearn.metrics import r2_score

y_true = Y_test.values.flatten()
y_pred = pred_test.flatten()

mask = y_true != 0
mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

r2 = r2_score(y_true, y_pred)

naive = Y_test.shift(1).iloc[1:].values.flatten()
true_aligned = Y_test.iloc[1:].values.flatten()
mae_naive = np.mean(np.abs(true_aligned - naive))
mae_model = mean_absolute_error(true_aligned, y_pred[1:])
mase = mae_model / mae_naive

true_diff = np.sign(np.diff(y_true))
pred_diff = np.sign(np.diff(y_pred))
directional_accuracy = np.mean(true_diff == pred_diff) * 100

print("\n=== FINAL METRICS ===")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")
print(f"MASE: {mase:.4f}")
print(f"DA:   {directional_accuracy:.2f}%")
