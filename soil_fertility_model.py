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
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Scale numeric columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Encode target column
le = LabelEncoder()
data['Output'] = le.fit_transform(data['Output'])

# Save mapping from numeric label -> original class
fertility_mapping = {i: cls for i, cls in enumerate(le.classes_)}
joblib.dump(fertility_mapping, "fertility_mapping.pkl")

print("Class distribution in dataset:")
print(data['Output'].value_counts())
print("Fertility mapping:", fertility_mapping)

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
# 5. Train Random Forest model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# =========================
# 6. Evaluate model
# =========================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save instead of showing
plt.close()

# Feature Importance
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Soil Nutrients)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig("feature_importance.png")  # Save instead of showing
plt.close()

# =========================
# 7. Save model, scaler, label encoder
# =========================
joblib.dump(model, "soil_fertility_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model, scaler, label encoder, and mapping saved successfully!")