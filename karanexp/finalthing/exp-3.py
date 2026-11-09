
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = load_diabetes(as_frame=True).frame
X, y = df.drop('target', axis=1).values, df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5),
    'Lasso': LassoCV(alphas=np.logspace(-3, 2, 50), cv=5,
                     max_iter=10000)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append([name, mean_squared_error(y_test, pred),
                    mean_squared_error(y_test, pred)**0.5,
                    r2_score(y_test, pred)])
    if name != 'Linear':
        print(name, 'Best alpha:', model.alpha_)

results = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])
print(results)

# Compare coefficients
names = df.drop('target', axis=1).columns
x = np.arange(len(names))
plt.bar(x-0.25, models['Linear'].coef_, 0.25, label='Linear')
plt.bar(x, models['Ridge'].coef_, 0.25, label='Ridge')
plt.bar(x+0.25, models['Lasso'].coef_, 0.25, label='Lasso')
plt.xticks(x, names, rotation=45)
plt.ylabel('Coefficient')
plt.legend()
plt.tight_layout()
plt.show()

# Compare R2 scores
plt.bar(results['Model'], results['R2'])
plt.ylabel('R2 Score')
plt.title('Model Performance Comparison')
plt.show()
