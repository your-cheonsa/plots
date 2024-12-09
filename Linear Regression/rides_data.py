# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:38:27 2024

@author: MSI
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data into a Pandas DataFrame
data = pd.read_csv('rides_data2.csv')

# Convert the 'Date' column to datetime
data['Date_Occurred'] = pd.to_datetime(data['Date'], errors='coerce')

# Create a 'Week' column representing the week of the year
data['Week'] = data['Date_Occurred'].dt.to_period('W')

# Group data by 'Week' and calculate total revenue (Fare) for each week
weekly_revenue = data.groupby('Week')['Fare'].sum().reset_index()

# Convert 'Week' to string for proper plotting
weekly_revenue['Week'] = weekly_revenue['Week'].astype(str)

# Create numerical week index (week number for regression)
weekly_revenue['Week_Number'] = range(len(weekly_revenue))

# Prepare the features (X) and target (y)
X = weekly_revenue[['Week_Number']]  # Week number as the feature
y = weekly_revenue['Fare']           # Total revenue as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) of the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Add the predictions to the original data (for plotting purposes)
weekly_revenue['Predicted_Fare'] = model.predict(weekly_revenue[['Week_Number']])

# Plot the revenue per week and the predictions
plt.figure(figsize=(12, 6))
sns.lineplot(x='Week', y='Fare', data=weekly_revenue, marker='o', label='Actual Revenue')
sns.lineplot(x='Week', y='Predicted_Fare', data=weekly_revenue, marker='x', label='Predicted Revenue')

# Add titles and labels
plt.title('Revenue per Week with Linear Regression Predictions', fontsize=16)
plt.xlabel('Week (Year-Week Number)', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

# Ensure layout is tight
plt.tight_layout()

# Show the plot
plt.legend()
plt.show()
