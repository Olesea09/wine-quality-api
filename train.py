"""
Script de antrenare a modelului Random Forest pentru Wine Quality.
Se rulează o singură dată: descarcă dataset-ul, antrenează modelul,
evaluează performanța și salvează modelul în model.pkl.

Sursa dataset-ului: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/186/wine+quality
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ============================================================
# 1. ÎNCĂRCAREA DATELOR
# ============================================================
# Folosim dataset-ul Red Wine Quality de la UCI
# 11 caracteristici fizico-chimice + 1 coloană target (quality: 0-10)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

print("Descarc dataset-ul Wine Quality (red) de la UCI...")
df = pd.read_csv(URL, sep=";")
print(f"Dataset incarcat: {df.shape[0]} randuri, {df.shape[1]} coloane")
print(f"Coloane: {list(df.columns)}")

# ============================================================
# 2. PREGĂTIREA DATELOR
# ============================================================
# Separăm caracteristicile (X) de target (y)
# X = cele 11 caracteristici fizico-chimice
# y = scorul de calitate (3-8 în practică)

X = df.drop("quality", axis=1)
y = df["quality"]

print(f"\nDistributia calitatii in dataset:")
print(y.value_counts().sort_index())

# Împărțim datele în train (80%) și test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  # pentru reproductibilitate
    stratify=y         # păstrăm distribuția claselor
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# ============================================================
# 3. ANTRENAREA MODELULUI
# ============================================================
# Random Forest: ensemble de arbori de decizie
# - n_estimators=100: 100 de arbori
# - random_state=42: rezultate reproductibile

print("\nAntrenez modelul Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # foloseste toate core-urile CPU disponibile
)
model.fit(X_train, y_train)
print("Antrenare finalizata.")

# ============================================================
# 4. EVALUAREA MODELULUI
# ============================================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAcuratete pe setul de test: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nRaport detaliat:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# 5. SALVAREA MODELULUI
# ============================================================
# joblib este mai eficient decat pickle pentru obiecte scikit-learn

MODEL_PATH = "model.pkl"
joblib.dump(model, MODEL_PATH)
print(f"Model salvat in: {MODEL_PATH}")

# Salvam si lista de feature names (utilă pentru API)
FEATURES_PATH = "features.pkl"
joblib.dump(list(X.columns), FEATURES_PATH)
print(f"Feature names salvate in: {FEATURES_PATH}")

print("\nGata. Modelul este gata de utilizare in API.")
