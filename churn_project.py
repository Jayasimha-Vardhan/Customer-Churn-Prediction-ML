import pandas as pd
import numpy as np

df = pd.read_csv("Telco-Customer-Churn.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nData Types:\n", df.dtypes)

# Churn distribution
print("\nChurn Value Counts:\n", df["Churn"].value_counts())

print("\nChurn Percentage:\n", df["Churn"].value_counts(normalize=True) * 100)

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()


sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()


sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()


df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(exclude="object").columns

print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)
df = pd.get_dummies(df, drop_first=True)
print(df.shape)

from sklearn.model_selection import train_test_split

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)

# Train First ML Model (Logistic Regression)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Train Second ML Model (Random Forest Model)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Handling Class Imbalance with Balanced Logistic Regression

model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
model_bal.fit(X_train, y_train)

y_pred_bal = model_bal.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

print("Balanced Logistic Accuracy:", accuracy_score(y_test, y_pred_bal))
print("\nClassification Report:\n", classification_report(y_test, y_pred_bal))

# Feature Importance from Balanced Logistic Regression

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model_bal.coef_[0]
})

importance["Abs"] = importance["Coefficient"].abs()
importance = importance.sort_values("Abs", ascending=False)

print(importance.head(10))

# Retention Strategy Recommendations


def retention_recommendation(row):
    recs = []

    if row.get("Contract_Two year",0)==0 and row.get("Contract_One year",0)==0:
        recs.append("Offer long-term contract discount")

    if row.get("OnlineSecurity_Yes",0)==0:
        recs.append("Offer free online security trial")

    if row.get("TechSupport_Yes",0)==0:
        recs.append("Provide priority tech support")

    if row.get("PaymentMethod_Electronic check",0)==1:
        recs.append("Suggest auto bank payment option")

    if row.get("InternetService_Fiber optic",0)==1:
        recs.append("Provide fiber service discount")

    return ", ".join(recs)

# Apply recommendations to predicted churned customers

X_test_copy = X_test.copy()
X_test_copy["Predicted_Churn"] = y_pred_bal

X_test_copy["Retention_Action"] = X_test_copy.apply(retention_recommendation, axis=1)

print(X_test_copy[["Predicted_Churn","Retention_Action"]].head(10))






