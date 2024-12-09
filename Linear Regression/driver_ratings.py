# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:38:27 2024

@author: MSI
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data into a Pandas DataFrame
data = pd.read_csv('rides_data2.csv')

# Preview the first few rows of the dataset
print(data.head())


# Preprocessing: Create a pipeline to encode categorical features and scale numerical features
# Separate features and target variable
X = data[['Distance_km', 'Duration_min', 'City']]  # Features (Distance, Duration, City)
y = data['Rating']  # Target variable (Driver Rating)

# Create a Column Transformer to apply OneHotEncoder to 'City' and StandardScaler to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Distance_km', 'Duration_min']),
        ('cat', OneHotEncoder(), ['City'])
    ])

# Create a Pipeline that first preprocesses the data and then fits a regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict ratings on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Plot the true vs predicted ratings
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Ratings')
plt.ylabel('Predicted Ratings')
plt.title('True vs Predicted Driver Ratings')
plt.show()

# Optionally, visualize the importance of features in the model
# For Linear Regression, we can check the coefficients (weights of each feature)
coefficients = model.named_steps['regressor'].coef_
print("Feature Coefficients (Weights):")
print(coefficients)
