
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
degrees, results = [1, 2, 3, 4, 6, 8], []
for d in degrees:
    model = make_pipeline(PolynomialFeatures(degree=d),
                          LinearRegression())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse, r2 = mean_squared_error(y_test, pred), r2_score(y_test, pred)
    results.append([d, mse, mse**0.5, r2])
    print(f'Degree {d}: MSE={mse:.4f}, RMSE={mse**0.5:.4f}, R2={r2:.4f}')

# Results and graph
results = pd.DataFrame(results, columns=['Degree', 'MSE', 'RMSE', 'R2'])
print(results)

plt.plot(results['Degree'], results['R2'], marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R2 Score')
plt.title('Polynomial Regression Comparison')
plt.grid()
plt.tight_layout()
plt.show()