# %% Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# %% Read the data from the CSV file
path = r'C:\Users\hp\Desktop\newproject\ssproject\Scripts\churnprediction\data\Bank_churn.csv'
df = pd.read_csv(path)

# Remove the rownumber, customerid columns, and surname
df = df.drop(['rownumber', 'customerid', 'surname'], axis=1)

# %% Data Overview
print(df.head())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# %% Visualizations
# Churn distribution
sns.countplot(x='churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Correlation matrix for numerical features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Plot distribution of numerical features
df.select_dtypes(include=['int64', 'float64']).hist(bins=20, figsize=(15, 10))
plt.suptitle('Numerical Feature Distributions')
plt.show()

# Distribution of categorical features
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=feature, data=df)
    plt.title(f'Distribution of {feature}')
    plt.show()

# %% Feature Engineering
# Age Binning
df['age_bin'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

# Balance Binning
df['balance_bin'] = pd.cut(df['balance'], bins=[-np.inf, 0, 50000, 100000, np.inf], labels=['Negative', 'Low', 'Medium', 'High'])

# Tenure Binning
df['tenure_bin'] = pd.cut(df['tenure'], bins=[-np.inf, 1, 3, 5, 7, 10, np.inf], labels=['0-1', '1-3', '3-5', '5-7', '7-10', '10+'])

# Encoding Number of Products
df['numofproducts_cat'] = df['numofproducts'].astype('category')

# Salary Binning
df['salary_bin'] = pd.cut(df['estimatedsalary'], bins=[-np.inf, 30000, 70000, 100000, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])

# %% Preprocessing
X = df.drop('churn', axis=1)
y = df['churn']

# Define steps for preprocessing numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())])  # Standardize features

# Define steps for preprocessing categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])  # One-hot encoding

# Combine both transformers into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Save the preprocessor to a pickle file
joblib.dump(preprocessor, 'preprocessor.pkl')

# %% Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_pred_log))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_pred_rf))

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_xgb))

# SVM
svm_model = SVC(probability=True, kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("\nSVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("ROC AUC:", roc_auc_score(y_test, y_pred_svm))

# Save the models to pickle files
joblib.dump(log_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(svm_model, 'svm_model.pkl')
    