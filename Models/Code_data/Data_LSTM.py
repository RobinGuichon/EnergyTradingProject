import pandas as pd
import numpy as np

# =========================================================
# 1) Chargement du dataset complet
# =========================================================
FILE_PATH = "/Users/robinguichon/Desktop/ProjetEnerg/DATA_UTILES/DATA0/dataset_merged_final.csv"
df = pd.read_csv(FILE_PATH)

print("Colonnes du dataset :")
print(df.columns)

# =========================================================
# 2) Vérifier que la colonne à supprimer existe
# =========================================================
col_to_drop = "forecast wind offshore eday ahead"

if col_to_drop not in df.columns:
    print(f"⚠️ Attention : la colonne '{col_to_drop}' n'existe pas dans le dataset !")
else:
    df = df.drop(columns=[col_to_drop])
    print(f"✔️ Colonne supprimée : {col_to_drop}")

# =========================================================
# 3) Gestion des valeurs manquantes
#    Interpolation linéaire + back/forward fill
# =========================================================
df = df.interpolate(method='linear', limit_direction='both')
df = df.fillna(method='bfill')
df = df.fillna(method='ffill')

print("\nPourcentage de valeurs manquantes après nettoyage :")
print(df.isna().mean() * 100)

# =========================================================
# 4) Séparation X / Y
#    IMPORTANT : on garde la colonne 'time' dans X
# =========================================================
TARGET = "price actual"

if TARGET not in df.columns:
    raise ValueError(f"❌ La colonne '{TARGET}' (target) n'existe pas dans le dataset.")

Y = df[TARGET].copy()

# X = toutes les colonnes sauf la target → incluant 'time'
X = df.drop(columns=[TARGET]).copy()

print(f"\n✔️ X shape : {X.shape}")
print(f"✔️ Y shape : {Y.shape}")

# =========================================================
# 5) Sauvegarde des fichiers
# =========================================================
X_OUTPUT = "X_clean_LSTM.csv"
Y_OUTPUT = "Y_clean_LSTM.csv"

X.to_csv(X_OUTPUT, index=False)
Y.to_csv(Y_OUTPUT, index=False)

print("\nFichiers générés :")
print(f" - {X_OUTPUT}")
print(f" - {Y_OUTPUT}")
