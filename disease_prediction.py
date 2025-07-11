# Disease Prediction from Medical Data (Diabetes Prediction)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
data = pd.read_csv('diabetes.csv')  # Make sure this file is in the same folder
print("✅ Dataset Loaded Successfully!")
print(data.head())

# Step 3: Prepare Data
X = data.drop('Outcome', axis=1)  # Features (input)
y = data['Outcome']               # Target (output)

# Step 4: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Standardize the Data (Better Model Performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build and Train the Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Test the Model
y_pred = model.predict(X_test)

# Step 8: Display Results
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Predict Example Data (Fixed Warning by Using DataFrame)
sample_input = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]],
                            columns=X.columns)
sample_input = scaler.transform(sample_input)
prediction = model.predict(sample_input)
print("\nPrediction for Sample Input (0 = No Diabetes, 1 = Diabetes):", prediction[0])
