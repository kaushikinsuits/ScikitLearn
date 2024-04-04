import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(0)
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)

linear_regression = LinearRegression()

linear_regression.fit(X, y)

X_new = np.array([[0], [2]])  
predictions = linear_regression.predict(X_new)

print("Coefficients:", linear_regression.coef_)
print("Intercept:", linear_regression.intercept_)

print("Predictions for new data points:")
for i, x in enumerate(X_new):
    print("X =", x[0], "-> Prediction =", predictions[i][0])
