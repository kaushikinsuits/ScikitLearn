# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Feature matrix
y = np.array([2, 4, 5, 4, 5])  # Target vector

# Creating a Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X, y)

# Making predictions
X_test = np.array([[6], [7], [8]])  # New data points
predictions = model.predict(X_test)

# Printing predictions
print("Predictions:", predictions)
