import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array([[1.0], [2.0], [3.0], [4.0]])  # Sample data

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nScaled data:")
print(scaled_data)
