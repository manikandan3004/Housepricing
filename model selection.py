import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load your dataset
df = pd.read_csv('Cleaned_Housing.csv')  # Replace 'your_dataset.csv' with the actual file path

# Inspect the dataset
print(df.head())
print(df.info())
print(df['price'].value_counts())  # Replace 'target' with the actual target column name

# Example data and target columns (Replace 'target' with your actual target column name)
X = df.drop('area', axis=1)
y = df['stories']

# Check for missing values and handle them (e.g., imputation)
X = X.fillna(X.mean())

# Remove the least populated class if it has only 1 member
class_counts = y.value_counts()
least_populated_classes = class_counts[class_counts == 1].index
X = X[~y.isin(least_populated_classes)]
y = y[~y.isin(least_populated_classes)]

# Ensure the test size is a valid float between 0 and 1
test_size_ratio = max(0.2, len(y) / (2 * len(y.unique())))
test_size_ratio = min(test_size_ratio, 0.5)  # Ensure it does not exceed 0.5 for practical purposes

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_train_resampled))

# Define a pipeline with scaling and the classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Define the parameter grid
param_grid = [
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_features': ['auto', 'sqrt', 'log2']
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
]

# Use StratifiedKFold to maintain class distribution
cv = StratifiedKFold(n_splits=2)  # Adjusting to fewer splits due to the small number of samples per class

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', error_score=np.nan)
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test set accuracy: {accuracy_score(y_test, y_pred)}")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
