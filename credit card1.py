# Credit Card Fraud Detection (Mini Project)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Step 1: Load Data
# -----------------------------
df = pd.read_csv("credit_card_sample.csv")
print("🔹 Dataset loaded. First 5 rows:")
print(df.head())

# -----------------------------
# Step 2: Basic Info
# -----------------------------
print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Class Distribution (0 = Normal, 1 = Fraud):")
print(df['Class'].value_counts())

# -----------------------------
# Step 3: Preprocessing
# -----------------------------v  
# Just check columns
print("\n🔹 Columns Present:", df.columns.tolist())

# Features & Target
X = df.drop('Class', axis=1)
y = df['Class']

# -----------------------------
# Step 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Step 5: Logistic Regression (Supervised)
# -----------------------------
print("\n🔹 Logistic Regression Training...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Evaluation
# -----------------------------
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Step 7: Isolation Forest (Unsupervised)
# -----------------------------
print("\n🔹 Isolation Forest - Anomaly Detection")
iso = IsolationForest(contamination=0.005, random_state=42)
y_pred_iso = iso.fit_predict(X)

# Map predictions: 1 → 0 (normal), -1 → 1 (fraud)
df['Anomaly'] = pd.Series(y_pred_iso).map({1: 0, -1: 1})

print("\n🔹 Isolation Forest - Detected Anomalies:")
print(df['Anomaly'].value_counts())

# -----------------------------
# Done
# -----------------------------
print("\n✅ Done. Both Logistic Regression and Isolation Forest ran successfully.")