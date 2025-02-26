# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load dataset
# df = pd.read_csv("Loan-Approval-Prediction.csv", encoding='utf-8')


# # Handle missing values
# imputer = SimpleImputer(strategy='mean')
# df[['ApplicantIncome', 'LoanAmount']] = imputer.fit_transform(df[['ApplicantIncome', 'LoanAmount']])

# df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical missing values with mode

# # Convert categorical variables to numerical
# df = pd.get_dummies(df, drop_first=True)

# # Normalize numerical features
# scaler = StandardScaler()
# df[['ApplicantIncome', 'LoanAmount']] = scaler.fit_transform(df[['ApplicantIncome', 'LoanAmount']])

# # Define features and target
# X = df.drop(columns=['Loan_Status'])  # Assuming 'Loan_Status' is the target
# y = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Convert to binary

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define models
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier()
# }

# # Train and evaluate models
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f"{name} Performance:")
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#     print("-" * 50)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_csv("Loan-Approval-Prediction.csv")

# Debug: Check column names
print("Columns in dataset:", df.columns)

# Handle missing values
categorical_data = ["Gender", "Married", "Self_Employed", "Dependents"]
for col in categorical_data:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

# Drop duplicates
df.drop_duplicates(inplace=True)

# Normalize numerical features
df["ApplicantIncome"] = (df["ApplicantIncome"] - df["ApplicantIncome"].mean()) / df["ApplicantIncome"].std()
df["LoanAmount"] = (df["LoanAmount"] - df["LoanAmount"].mean()) / df["LoanAmount"].std()

# Check if 'Loan_Status' column exists
if 'Loan_Status' not in df.columns:
    raise KeyError("Loan_Status column not found in dataset")

# Define features and target
X = df.drop(columns=['Loan_Status', 'Loan_ID'], errors='ignore')  # Ignore if 'Loan_ID' doesn't exist
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Encode categorical features
encoder = LabelEncoder()
for col in ["Education", "Married", "Property_Area", "Gender", "Self_Employed", "Dependents"]:
    if col in X.columns:
        X[col] = encoder.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save and load model
joblib.dump(model, "loan-approval-predictor.pkl")
loaded_model = joblib.load("loan-approval-predictor.pkl")

# Make predictions
y_prediction = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_prediction) * 100

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_prediction))
print("Classification Report:\n", classification_report(y_test, y_prediction))
