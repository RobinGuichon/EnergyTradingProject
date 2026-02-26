import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 1. Load Data ===
path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA2_XGBOOST"

X = pd.read_csv(f"{path}/X_clean_rolling_v2.csv")
Y = pd.read_csv(f"{path}/Y_clean_v2.csv")

# Convert time to datetime
X["time"] = pd.to_datetime(X["time"])

# Remove time from features
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

# === 3. DMatrix ===
dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_val, label=Y_val)

# === 4. Hyperparameters ===
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

print("\nTraining complete.")
print("Best iteration:", model.best_iteration)

# === 6. SAVE MODEL ===
model.save_model("xgb_intraday_model.json")

print("\n✔ Model saved as xgb_intraday_model.json")
