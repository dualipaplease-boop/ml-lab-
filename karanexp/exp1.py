
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = fetch_california_housing(as_frame=True).frame
X, y = df[['AveRooms']].values, df['MedHouseVal'].values
mask = (X.flatten() < 10) & (y < 5)
X, y = X[mask], y[mask]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred1 = lr.predict(X_test)
print('Linear Regression MSE:', mean_squared_error(y_test, pred1))
print('Linear Regression R2:', r2_score(y_test, pred1))

# SGD Regressor
sgd = SGDRegressor(random_state=42)
sgd.fit(X_train, y_train)
pred2 = sgd.predict(X_test)
print('SGD MSE:', mean_squared_error(y_test, pred2))
print('SGD R2:', r2_score(y_test, pred2))

# Plot
plt.scatter(X_test, y_test, s=8, alpha=0.4, label='Actual')
plt.plot(X_test, pred1, '.', label='Linear Regression')
plt.plot(X_test, pred2, '.', label='SGD Regressor')
plt.xlabel('AveRooms (Scaled)')
plt.ylabel('Median House Value')
plt.legend()
plt.tight_layout()
plt.show()
