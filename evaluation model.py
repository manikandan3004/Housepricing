import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'Cleaned_Housing.csv'
housing_data = pd.read_csv(file_path)

# Split the data into features and target variable
X = housing_data.drop(columns=['price'])
y = housing_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Model RMSE: {rmse}")

def evaluate_house_price(features):
    # Create a DataFrame with the user input
    feature_names = X.columns
    user_input_df = pd.DataFrame([features], columns=feature_names)
    
    # Convert user input into the appropriate format
    user_input_scaled = scaler.transform(user_input_df)
    
    # Predict the house price using the model
    predicted_price = model.predict(user_input_scaled)[0]
    
    # Define a threshold for evaluation (e.g., average price in the dataset)
    average_price = np.mean(y)
    
    # Evaluate if the price is worth it
    if predicted_price <= average_price:
        result = "The price of the house is worth it."
    else:
        result = "The price of the house is not worth it."
    
    return predicted_price, result

# Get user input
area = float(input("Enter the area (in square feet): "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))
stories = int(input("Enter the number of stories: "))
mainroad = int(input("Is the house on the main road? (1: Yes, 0: No): "))
guestroom = int(input("Does the house have a guest room? (1: Yes, 0: No): "))
basement = int(input("Does the house have a basement? (1: Yes, 0: No): "))
hotwaterheating = int(input("Does the house have hot water heating? (1: Yes, 0: No): "))
airconditioning = int(input("Does the house have air conditioning? (1: Yes, 0: No): "))
parking = int(input("Enter the number of parking spaces: "))
prefarea = int(input("Is the house in a preferred area? (1: Yes, 0: No): "))
furnished = int(input("Is the house furnished? (1: Yes, 0: No): "))
semi_furnished = int(input("Is the house semi-furnished? (1: Yes, 0: No): "))
unfurnished = int(input("Is the house unfurnished? (1: Yes, 0: No): "))

user_features = [
    area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
    hotwaterheating, airconditioning, parking, prefarea, furnished,
    semi_furnished, unfurnished
]

# Evaluate the user input
predicted_price, evaluation_result = evaluate_house_price(user_features)
print(f"Predicted Price: {predicted_price}")
print(evaluation_result)
