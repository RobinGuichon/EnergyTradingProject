import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
#                CONFIGURATION
# ============================================================

WINDOW = 72  # 72 heures d'historique
path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA1_LSTM"

print("\n==============================")
print("     LSTM FULL PIPELINE")
print("==============================\n")


# ============================================================
#                1. LOAD + PREPROCESS
# ============================================================

print("1) Chargement des données...")

X = pd.read_csv(f"{path}/X_clean_LSTM.csv")
Y = pd.read_csv(f"{path}/Y_clean_LSTM.csv")

# Convert time to datetime
time = pd.to_datetime(X["time"])

# Remove time column for model input
X = X.drop(columns=["time"])

# Normalize
print("Normalisation...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, f"{path}/lstm_scaler.pkl")

X_arr = np.array(X_scaled)
Y_arr = Y.values.flatten()


# ============================================================
#                2. BUILD SEQUENCES (72h)
# ============================================================

print("Construction des séquences (72h)...")

X_seq = []
Y_seq = []

for t in range(WINDOW, len(X_arr)):
    X_seq.append(X_arr[t-WINDOW:t])
    Y_seq.append(Y_arr[t])

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)

print("Shape séquences :", X_seq.shape)  # (N, 72, features)


# ============================================================
#                3. TEMPORAL SPLIT
# ============================================================

N = len(X_seq)
train_size = int(0.70 * N)
val_size = int(0.15 * N)

X_train = X_seq[:train_size]
Y_train = Y_seq[:train_size]

X_val = X_seq[train_size:train_size + val_size]
Y_val = Y_seq[train_size:train_size + val_size]

X_test = X_seq[train_size + val_size:]
Y_test = Y_seq[train_size + val_size:]

time_seq = time[WINDOW:]  # realigned time

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# ============================================================
#                4. MODEL LSTM
# ============================================================

print("\n2) Construction du modèle LSTM...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, X_train.shape[2])),
    Dropout(0.2),

    LSTM(32),
    Dropout(0.2),

    Dense(1)
])

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

model.summary()


# ============================================================
#                5. TRAINING
# ============================================================

print("\n3) Entraînement du modèle...")

es = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=80,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# Save model + history
model.save(f"{path}/lstm_72h_model.keras")
joblib.dump(history.history, f"{path}/lstm_72h_history.pkl")

print("\n✔ Entraînement terminé.")


# ============================================================
#                6. PREDICTIONS + METRICS
# ============================================================

print("\n4) Évaluation sur le test set...")

y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
mae = mean_absolute_error(Y_test, y_pred)
mape = np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100
r2 = r2_score(Y_test, y_pred)

true_diff = np.sign(np.diff(Y_test))
pred_diff = np.sign(np.diff(y_pred))
DA = (true_diff == pred_diff).mean() * 100

# MASE
naive = Y_test[:-1]
mae_naive = np.mean(np.abs(Y_test[1:] - naive))
MASE = mae / mae_naive

print("\n=== METRICS ===")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print(f"DA   : {DA:.2f}%")
print(f"MASE : {MASE:.4f}")


# ============================================================
#                7. PLOT RESULTS — CUSTOM STYLE
# ============================================================

print("\n5) Affichage des prédictions...")

import matplotlib as mpl

# --- Global Times New Roman style ---
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

# --- Custom Colors (normalized RGB) ---
color_real = (103/255, 16/255, 9/255)       # dark red/brown
color_pred = (231/255, 187/255, 65/255)     # gold

time_test_plot = time_seq[-len(Y_test):]

plt.figure(figsize=(12, 6))
plt.plot(time_test_plot, Y_test, label="Real Price",
         linewidth=0.8, color=color_real)
plt.plot(time_test_plot, y_pred, label="Predicted Price",
         linewidth=0.8, color=color_pred)

plt.grid(True)
plt.title("LSTM_72h — Real vs Predicted Electricity Price (Test Set)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()

# --- Save figure ---
save_path = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/lstm_72h_test.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✔ Test figure saved at:\n{save_path}")

 # ============================================================
#                8. LAST MONTH FIGURE
# ============================================================

# Extract last 30 days (≈720 hours)
last_month_hours = 24 * 7

X_last = time_test_plot[-last_month_hours:]
Y_last = Y_test[-last_month_hours:]
pred_last = y_pred[-last_month_hours:]

plt.figure(figsize=(12, 6))
plt.plot(X_last, Y_last, label="Real Price",
         linewidth=1.2, color=color_real)
plt.plot(X_last, pred_last, label="Predicted Price",
         linewidth=1.2, color=color_pred)

plt.grid(True)
plt.title("LSTM_72h — Real vs Predicted Electricity Price (Last Week)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()

# --- Save figure ---
save_path_last = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/lstm_72h_last_month.png"
plt.savefig(save_path_last, dpi=300, bbox_inches="tight")
plt.show()

print(f"✔ Last-month figure saved at:\n{save_path_last}")

