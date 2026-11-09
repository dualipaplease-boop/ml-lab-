import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('../../data/auto-mpg-dataset.csv')
print(df.head())
print(df.describe())

# Visualize data
plt.hist(df['mpg'].dropna(), bins=30)
plt.xlabel('MPG')
plt.show()

plt.scatter(df['displacement'], df['mpg'])
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.show()

# Prepare data
df = df.dropna(subset=['displacement', 'mpg'])
X, y = df[['displacement']].values, df['mpg'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)

# Linear baseline
lin = LinearRegression().fit(X_train, y_train)
pred = lin.predict(X_test)
print('Linear:', mean_squared_error(y_test, pred),
      r2_score(y_test, pred))

# Polynomial regression
degrees, results, models = [1, 2, 3, 4, 6, 8], [], {}

for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d),
                          LinearRegression())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append([d, mse, mse**0.5, r2])
    models[d] = model

    print(f'Degree {d}: MSE={mse:.4f}, RMSE={mse**0.5:.4f}, R2={r2:.4f}')

# Results table
results = pd.DataFrame(results, columns=['Degree', 'MSE', 'RMSE', 'R2'])
print(results)

# R2 comparison graph
plt.plot(results['Degree'], results['R2'], marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.title('Polynomial Regression Comparison')
plt.grid()
plt.tight_layout()
plt.show()

# Visualization of fitted curves
xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(9, 6))
plt.scatter(X_test, y_test, alpha=0.4, s=15, label='Test Data')

plt.plot(xs, lin.predict(xs), 'k--', lw=2, label='Linear (Baseline)')

for d, color in zip([2, 4, 8], ['crimson', 'green', 'orange']):
    plt.plot(xs, models[d].predict(xs), color=color, lw=2,
             label=f'Degree {d}')

plt.xlabel('Engine Displacement (cu. in.)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Polynomial Regression Fits vs Linear Baseline')
plt.legend()
plt.tight_layout()
plt.show()