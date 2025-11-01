import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(r'C:\Users\prana\Desktop\API evelopment\backend_projects\WA_Fn-UseC_-Telco-Customer-Churn.csv')

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
categorical_features = data.select_dtypes(include=['object']).columns.drop('customerID')

model = LabelEncoder()
for i in categorical_features:
    data[i] = model.fit_transform(data[i])

X = data.drop(['customerID', 'Churn'], axis=1)
y = data['Churn']

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

print("Preprocessing completed.")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


ml_model = LogisticRegression()
ml_model.fit(X_train,y_train)
prediction = ml_model.predict(X_test)
accuracy = accuracy_score(y_test,prediction)
print(f'the model accuracy is: {accuracy}')
classification_Report = classification_report(y_test,prediction)
print(f'classification report is: {classification_report}')