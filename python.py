import pandas as pd
from joblib import dump , load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(42)


data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1) 
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
dump(model, "diabetes_model.pkl")  

y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)
print(f"Training Model Accuracy: {accuracy * 100:.2f}%")