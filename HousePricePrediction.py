# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to load and preprocess data
def load_data():
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['PRICE'] = boston.target
    return data

# Function to select features
def select_features(data, features):
    X = data[features]
    y = data['PRICE']
    return X, y

# Function to train models
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R-squared': r2}
    return results

# Function to save models
def save_models(models):
    for name, model in models.items():
        joblib.dump(model, f"{name}.joblib")

# Function to load models
def load_models(model_names):
    models = {}
    for name in model_names:
        models[name] = joblib.load(f"{name}.joblib")
    return models

# Main function
def main():
    data = load_data()
    features = ['RM', 'LSTAT', 'PTRATIO']  # Example features
    X, y = select_features(data, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    for name, result in results.items():
        print(f"{name}:")
        print(f"\tMSE: {result['MSE']}")
        print(f"\tR-squared: {result['R-squared']}")
    save_models(models)
    loaded_models = load_models(models.keys())
    print("Models saved and loaded successfully!")

if __name__ == "__main__":
    main()
