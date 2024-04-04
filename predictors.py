import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(0)
X_train = 2 * np.random.rand(100, 1)  # Generate 100 random numbers between 0 and 2 for training data
y_train = 4 + 3 * X_train + np.random.randn(100, 1)  # Linear relationship with some noise for training data

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

X_new = np.array([[0], [2]]) 

predictions = linear_regression.predict(X_new)

print("Predictions for new data points:")
for i, x in enumerate(X_new):
    print("X =", x[0], "-> Prediction =", predictions[i][0])
