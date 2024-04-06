# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define a pipeline for training and testing
pipelines = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=100))]),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
}

# Hyperparameters for tuning
params = {
    'Logistic Regression': {'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0]},
    'Random Forest': {'n_estimators': [50, 100, 200]},
    'SVM': {'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0], 'clf__gamma': [0.001, 0.01, 0.1, 1.0]}
}

# Train and evaluate models
results = {}
for name, pipe in pipelines.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(pipe, params[name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'best_model': best_model, 'accuracy': accuracy}
    print(f"{name} Accuracy: {accuracy:.4f}")

# Print best accuracy and model details
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print("Best Model Parameters:")
print(results[best_model_name]['best_model'])

# Generate classification report and confusion matrix for best model
best_model = results[best_model_name]['best_model']
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize some misclassified digits
misclassified_idx = np.where(y_pred != y_test)[0]
num_samples = 5
plt.figure(figsize=(10, 4))
for i, idx in enumerate(misclassified_idx[:num_samples]):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {y_pred[idx]}, Actual: {y_test[idx]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
