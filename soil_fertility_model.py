import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load dataset
# =========================
data = pd.read_csv("dataset1.csv")

# =========================
# 2. Preprocessing
# =========================
numeric_cols = ['N','P','K','pH','EC','OC','S','Zn','Fe','Cu','Mn','B']

# Fill missing values with mean
for col in numeric_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Normalize numeric columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Encode target column
le = LabelEncoder()
data['Output'] = le.fit_transform(data['Output'])

# =========================
# 3. Features & target
# =========================
X = data[numeric_cols]
y = data['Output']

# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Train model
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# 6. Evaluate model
# =========================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 🔹 Feature Importance
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Soil Nutrients)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# =========================
# 7. Save model, scaler, and label encoder
# =========================
joblib.dump(model, "soil_fertility_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model, scaler, and label encoder saved.")

# =========================
# 8. Predict fertility for all rows in dataset
# =========================
# Keep original data for predictions
new_samples = pd.read_csv("dataset1.csv")
new_samples_features = new_samples[numeric_cols]

# Load saved objects
scaler = joblib.load("scaler.pkl")
model = joblib.load("soil_fertility_model.pkl")
le = joblib.load("label_encoder.pkl")

# Scale features and keep column names to avoid warnings
new_samples_scaled = scaler.transform(new_samples_features)
new_samples_scaled_df = pd.DataFrame(new_samples_scaled, columns=numeric_cols)

# Predict
predictions = model.predict(new_samples_scaled_df)
predicted_labels = le.inverse_transform(predictions)

# Add predictions to original dataframe
new_samples['Predicted_Fertility'] = predicted_labels

# Save predictions
new_samples.to_csv("predicted_soil_fertility.csv", index=False)
print("Predictions saved to 'predicted_soil_fertility.csv'.")
