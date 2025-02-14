import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

loaded_model = load("diabetes_model.pkl")

data = pd.read_csv("diabetes.csv") 
X = data.drop("Outcome", axis=1)  
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Loaded Model Accuracy: {accuracy * 100:.2f}%")