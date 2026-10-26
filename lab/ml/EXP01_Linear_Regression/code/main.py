import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

print("California Housing Dataset")
print(f"Shape: {X.shape}, {y.shape}")
print(f"Features: {feature_names}")
print(y[:5])

# Use only the first feature (MedInc - median income) for single-variable linear regression
X_single = X[:, :1]  # MedOnly

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nLinear Regression (Single Variable - MedIncome):")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='steelblue', alpha=0.6, label='Actual')
plt.plot(X_test, y_pred, color='crimson', linewidth=2, label='Predicted')
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value ($100k)')
plt.title('Linear Regression: Single Variable (MedInc) vs California Housing Price')
plt.legend()
plt.tight_layout()
plt.savefig('output/plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Save output
with open('output/output.txt', 'w') as f:
    f.write('California Housing Dataset - Linear Regression (Single Variable)\\n')
    f.write(f'Features used: {feature_names[0]}\\n')
    f.write(f'Coefficient: {model.coef_[0]:.4f}\\n')
    f.write(f'Intercept: {model.intercept_:.4f}\\n')
    f.write(f'Mean Squared Error: {mse:.4f}\\n')
    f.write(f'R2 Score: {r2:.4f}\\n')