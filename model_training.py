import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

import joblib


# Load dataset
data = pd.read_csv("dataset/flood_data.csv")

# Separate input and output
X = data.drop("flood", axis=1)
y = data["flood"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier()
}

best_model = None
best_accuracy = 0


# Train and compare models
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(name, "Accuracy:", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model


# Save best model
joblib.dump(best_model, "flood_model.pkl")

print("Best Model Accuracy:", best_accuracy)
print("Model saved successfully!")
