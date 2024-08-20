import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load your dataset
df = pd.read_csv('Cleaned_housing.csv')  # Replace 'your_housing_dataset.csv' with the actual file path

# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())

# Identify features and target variable (assuming 'price' is the target variable)
X = df.drop('price', axis=1)  # Replace 'price' with your actual target column name
y = df['price']  # Replace 'price' with your actual target column name

# Check for missing values and handle them (e.g., imputation)
X = X.fillna(X.mean())

# Ensure the test size is a valid float between 0 and 1
test_size_ratio = 0.2  # Typical test size for regression problems

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

# Define a pipeline with scaling and the regressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor())
])

# Define the parameter grid
param_grid = [
    {
        'regressor': [RandomForestRegressor()],
        'regressor__n_estimators': [10, 50, 100],
        'regressor__max_features': ['auto', 'sqrt', 'log2']
    },
    {
        'regressor': [SVR()],
        'regressor__C': [0.1, 1, 10],
        'regressor__kernel': ['linear', 'rbf']
    }
]

# Use KFold for cross-validation
cv = StratifiedKFold(n_splits=5)

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2', error_score=np.nan)
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print(f"Test set R^2 score: {r2_score(y_test, y_pred)}")
print(f"Test set RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Print additional evaluation details
print("\nDetailed Evaluation:")
print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred))}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")
