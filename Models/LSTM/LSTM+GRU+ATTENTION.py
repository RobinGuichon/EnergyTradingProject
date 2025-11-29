import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =====================================================
#  ATTENTION LAYER (simple, compatible CPU)
# =====================================================

class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(a * x, axis=1)

# =====================================================
# 1) LOAD DATA
# =====================================================

print("\n==============================")
print("     LSTM + GRU + ATTENTION")
print("==============================\n")

path = "/Users/robinguichon/Desktop/ProjetEnerg/Data/DATA1_LSTM"

X = pd.read_csv(f"{path}/X_clean_LSTM.csv")
Y = pd.read_csv(f"{path}/Y_clean_LSTM.csv")

X["time"] = pd.to_datetime(X["time"])
time_index = X["time"]
X = X.drop(columns=["time"])

# Remove rows where Y is NaN just in case
mask = ~Y["price actual"].isna()
X = X[mask]
Y = Y[mask]

# =====================================================
# 2) SCALING
# =====================================================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(Y)

SEQ_LEN = 72     # 72 hours

# =====================================================
# 3) BUILD SEQUENCES
# =====================================================

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

print("Construction des séquences...")
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

print("Shape séquences :", X_seq.shape)

# =====================================================
# 4) TRAIN / VAL / TEST SPLIT
# =====================================================

n = len(X_seq)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]

X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# =====================================================
# 5) BUILD MODEL : LSTM + GRU + ATTENTION
# =====================================================

print("\nConstruction du modèle LSTM+GRU+Attention...")

inputs = Input(shape=(SEQ_LEN, X_train.shape[2]))

x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)

x = GRU(32, return_sequences=True)(x)
x = Dropout(0.2)(x)

x = Attention()(x)

outputs = Dense(1)(x)

model = Model(inputs, outputs)

model.compile(loss="mse", optimizer="adam")

model.summary()

# =====================================================
# 6) TRAIN
# =====================================================

print("\nEntraînement du modèle...")

es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

print("\n✔ Entraînement terminé.")

# =====================================================
# 7) PREDICTIONS + INVERSE SCALING
# =====================================================

print("\nÉvaluation...")

y_pred_test = model.predict(X_test)
y_pred_test = scaler_y.inverse_transform(y_pred_test)
y_test_real = scaler_y.inverse_transform(y_test)

# =====================================================
# 8) METRICS
# =====================================================

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test))
mae = mean_absolute_error(y_test_real, y_pred_test)
mape = np.mean(np.abs((y_test_real - y_pred_test) / y_test_real)) * 100
r2 = r2_score(y_test_real, y_pred_test)

true_diff = np.sign(np.diff(y_test_real.flatten()))
pred_diff = np.sign(np.diff(y_pred_test.flatten()))
DA = np.mean(true_diff == pred_diff) * 100

# MASE
naive = y_test_real[:-1]
mae_naive = mean_absolute_error(y_test_real[1:], naive)
mase = mae / mae_naive

print("\n=== METRICS ===")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print(f"DA   : {DA:.2f}%")
print(f"MASE : {mase:.4f}")

# =====================================================
# 9) PLOT PRED VS REAL — CUSTOM STYLE
# =====================================================

import matplotlib as mpl

# --- Global Times New Roman style ---
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

# --- Custom Colors (normalized RGB) ---
color_real = (103/255, 16/255, 9/255)       # dark red
color_pred = (231/255, 187/255, 65/255)     # gold

# --- Build full time index for test set ---
time_test = time_index[-len(y_test_real):]

plt.figure(figsize=(12, 6))
plt.plot(time_test, y_test_real, label="Real Price",
         linewidth=0.8, color=color_real)
plt.plot(time_test, y_pred_test, label="Predicted Price",
         linewidth=0.8, color=color_pred)

plt.title("LSTM_GRU_Attention — Real vs Predicted Electricity Price (Test Set)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save figure ---
save_path = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/lstm_gru_att_test.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✔ Test-set figure saved at:\n{save_path}")

# =====================================================
# 10) LAST MONTH FIGURE
# =====================================================

# Approx last 30 days = 720 hours (adjust if needed)
last_month_hours = 24 * 7

time_last = time_test[-last_month_hours:]
y_last = y_test_real[-last_month_hours:]
pred_last = y_pred_test[-last_month_hours:]

plt.figure(figsize=(12, 6))
plt.plot(time_last, y_last, label="Real Price",
         linewidth=1.2, color=color_real)
plt.plot(time_last, pred_last, label="Predicted Price",
         linewidth=1.2, color=color_pred)

plt.title("LSTM_GRU_Attention — Real vs Predicted Electricity Price (Last Week)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Save figure ---
save_path_last = "/Users/robinguichon/Desktop/ProjetEnerg/Plots/lstm_gru_att_last_month.png"
plt.savefig(save_path_last, dpi=300, bbox_inches="tight")
plt.show()

print(f"✔ Last-month figure saved at:\n{save_path_last}")


