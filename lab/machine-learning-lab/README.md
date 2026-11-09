# 🤖 Machine Learning Lab

> **A practical, beginner-friendly guide to common machine learning lab tasks.**

Use this as a step-by-step reference for loading data, preparing features, training models, checking results, and creating plots with Python, Pandas, NumPy, Matplotlib, and scikit-learn.

---

## 🚀 Start Here

Most supervised learning experiments follow the same pattern:

```text
Load data → Inspect it → Choose features and target → Split → Preprocess → Train → Evaluate → Visualize
```

Start with [Universal ML Workflow](#1-universal-ml-workflow), then jump to the section for your model. Use the tables as a quick lookup during lab work or revision.

### At a glance

| If you need to… | Go to… |
|---|---|
| Load or inspect a dataset | [Dataset Loading](#3-dataset-loading-methods) · [Inspection](#4-dataset-display-and-inspection) |
| Prepare inputs for a model | [Preprocessing](#5-data-preprocessing) · [Features and Targets](#6-selecting-features-and-targets) · [Scaling](#8-feature-scaling) |
| Build a model | [Regression](#10-regression-experiments) · [Classification](#11-classification-experiments) · [Clustering](#12-clustering-experiments) |
| Check results or make plots | [Evaluation Metrics](#13-evaluation-metrics) · [Graphs](#14-common-graph-shortcuts-) |
| Prepare for an exam | [Common Mistakes](#17-common-mistakes) · [Viva Reference](#18-quick-viva-reference) |

---

## 📚 Table of Contents

1. [Universal ML Workflow](#1-universal-ml-workflow)
2. [Common Libraries](#2-common-libraries)
3. [Dataset Loading Methods](#3-dataset-loading-methods)
4. [Dataset Display and Inspection](#4-dataset-display-and-inspection)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Selecting Features and Targets](#6-selecting-features-and-targets)
7. [Train-Test Split](#7-train-test-split)
8. [Feature Scaling](#8-feature-scaling)
9. [Common Model Workflow](#9-common-model-workflow)
10. [Regression Experiments](#10-regression-experiments)
11. [Classification Experiments](#11-classification-experiments)
12. [Clustering Experiments](#12-clustering-experiments)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Common Graph Shortcuts](#14-common-graph-shortcuts)
15. [Useful Python and NumPy Shortcuts](#15-useful-python-and-numpy-shortcuts)
16. [Hyperparameter Tuning](#16-hyperparameter-tuning)
17. [Common Mistakes](#17-common-mistakes)
18. [Quick Viva Reference](#18-quick-viva-reference)

---

## 1. Universal ML Workflow

Use this checklist for most **supervised learning** practicals. The exact preprocessing and evaluation steps depend on the dataset and task:

**Load Data → Explore Data → Select X and y → Preprocess → Split Data → Scale → Train → Predict → Evaluate → Visualize**

### Step 1: Load the dataset

```python
df = pd.read_csv('data.csv')
```

### Step 2: Explore the dataset

```python
print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df.isnull().sum())
```

### Step 3: Select features and target

```python
X = df.drop('target', axis=1)
y = df['target']
```

- `X` contains the input features.
- `y` contains the target/output variable.

### Step 4: Preprocess the data

Depending on the experiment, this may include handling missing values, encoding categorical variables, removing outliers, or scaling features.

### Step 5: Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 6: Scale features if needed

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 7: Train the model

```python
model = SomeModel()
model.fit(X_train, y_train)
```

### Step 8: Make predictions

```python
pred = model.predict(X_test)
```

### Step 9: Evaluate the model

Use metrics appropriate to the experiment: MSE/RMSE/R² for regression or accuracy, precision, recall, and F1-score for classification.

### Step 10: Visualize the results

Use scatter plots, line plots, histograms, confusion matrices, or elbow graphs as required.

> **Clustering note:** Clustering is usually unsupervised. You typically do not need target labels or a train-test split.

---

## 2. Common Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

| Library | Purpose |
|---|---|
| NumPy (`np`) | Numerical calculations, arrays, and matrices |
| Pandas (`pd`) | Loading and manipulating tabular data |
| Matplotlib (`plt`) | Creating graphs |
| Scikit-learn (`sklearn`) | Preprocessing, models, and evaluation |

Import only the libraries needed for the experiment.

---

## 3. Dataset Loading Methods

### A. Load an existing CSV file

Use this when the dataset is saved in your project folder.

```python
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())
```

Example folder structure:

```text
ML-Lab/
├── experiment.py
├── data.csv
└── README.md
```

If the CSV is in a subfolder:

```python
df = pd.read_csv('data/data.csv')
```

Using a full Windows path:

```python
df = pd.read_csv(r'C:\Users\Student\Downloads\data.csv')
```

The `r` prefix creates a raw string and helps avoid backslash escape issues.

> **Tip:** Relative paths are resolved from the current working directory. If Python cannot find a file, check that directory and the file name.

### B. Load a CSV file directly from a URL

If the URL points directly to a CSV file, Pandas can usually read it without saving it manually.

```python
url = 'https://example.com/data.csv'
df = pd.read_csv(url)

print(df.head())
```

Example using a public Auto MPG dataset:

```python
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
df = pd.read_csv(url)
```

The URL must be accessible and return CSV data rather than an ordinary webpage.

### C. Download a CSV file and save it locally

```python
import requests
import pandas as pd

url = 'https://example.com/data.csv'

response = requests.get(url, timeout=30)
response.raise_for_status()

with open('data.csv', 'wb') as f:
    f.write(response.content)

df = pd.read_csv('data.csv')
```

This method requires the `requests` library and internet access.

### D. Upload a CSV file in Google Colab

```python
from google.colab import files
import pandas as pd

uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))

print(df.head())
```

This opens a file picker. Select the CSV file from your computer.

If the file is already uploaded:

```python
df = pd.read_csv('/content/data.csv')
```

The file must exist at that location.

### E. Load a CSV file from Google Drive

```python
from google.colab import drive
import pandas as pd

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/data.csv')
```

Replace the path with the actual location of the file in Google Drive.

### F. Load an Excel file

```python
df = pd.read_excel('data.xlsx')
```

To select a particular sheet:

```python
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

Reading Excel files may require an additional dependency such as `openpyxl`.

### G. Load built-in Scikit-learn datasets

**California Housing:**

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
```

**Diabetes:**

```python
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
df = data.frame
```

**Iris:**

```python
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
```

`load_*` datasets are generally packaged with Scikit-learn. `fetch_*` datasets may need to be downloaded or retrieved from a local cache on the first run.

### H. Load a built-in dataset directly as X and y

```python
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target
```

This is convenient when you do not need a DataFrame.

### I. Load a CSV from a ZIP archive

```python
df = pd.read_csv('data.zip', compression='zip')
```

This works when Pandas can identify the CSV inside the archive. For complex archives, extract the CSV first.

### Quick dataset-loading reference

| Source | Syntax |
|---|---|
| Local CSV | `pd.read_csv('data.csv')` |
| CSV URL | `pd.read_csv(url)` |
| Excel | `pd.read_excel('data.xlsx')` |
| Colab upload | `files.upload()` |
| Google Drive | `drive.mount('/content/drive')` |
| Built-in dataset | `load_iris()` |
| Downloadable dataset | `fetch_california_housing()` |

---

## 4. Dataset Display and Inspection

These commands are useful before training any ML model.

### A. Display the dataset

```python
print(df)          # Display DataFrame as text
display(df)        # Formatted table in Jupyter/Colab
print(df.head())   # First 5 rows
```

In Jupyter Notebook or Colab, you can also write `df.head()` as the last line of a cell to display the result automatically.

### B. Display the first and last rows

```python
df.head()       # First 5 rows
df.head(10)     # First 10 rows
df.tail()       # Last 5 rows
df.tail(3)      # Last 3 rows
```

### C. Display dataset dimensions and column information

```python
print(df.shape)    # Rows and columns
print(df.columns)  # Column names
print(df.dtypes)   # Data types
df.info()          # Column details and non-null counts
```

If `df.shape` returns `(1000, 8)`, there are 1,000 rows and 8 columns.

`df.info()` prints its report directly. Avoid `print(df.info())`, which adds an extra `None` line.

### D. Display statistical summary

```python
print(df.describe())
```

`describe()` displays summary statistics for numerical columns by default.

| Statistic | Meaning |
|---|---|
| `count` | Number of non-missing values |
| `mean` | Average |
| `std` | Standard deviation |
| `min` | Minimum |
| `25%` | First quartile |
| `50%` | Median |
| `75%` | Third quartile |
| `max` | Maximum |

For categorical columns:

```python
print(df.describe(include='object'))
```

For all columns:

```python
print(df.describe(include='all'))
```

### E. Display missing values

```python
print(df.isnull().sum())
```

Total missing values:

```python
print(df.isnull().sum().sum())
```

Check whether any missing values exist:

```python
print(df.isnull().values.any())
```

### F. Display individual columns

```python
print(df['Age'])                 # One column
print(df[['Age', 'Salary']])     # Multiple columns
print(df['Category'].unique())   # Unique values
print(df['Category'].value_counts())  # Frequency of each value
```

### G. Display specific rows and columns

```python
print(df.iloc[0])       # First row
print(df.iloc[0:5])     # First 5 rows
print(df.iloc[:, 0])    # First column
print(df.iloc[0, 1])    # First row, second column
```

Using column names and index labels:

```python
print(df.loc[0, 'Age'])
```

- `iloc` uses integer positions.
- `loc` uses index and column labels.

### H. Display dataset size

```python
print(len(df))       # Number of rows
print(df.shape[0])   # Number of rows
print(df.shape[1])   # Number of columns
print(df.size)       # Total number of cells
```

### I. Display NumPy arrays

```python
X = df[['Age']].values
y = df['Salary'].values

print(X)
print(y)
print(X.shape)
print(y.shape)
print(X.flatten())
```

- `.values` converts Pandas data into a NumPy array.
- `.shape` gives array dimensions.
- `.flatten()` converts an array into one dimension.

### J. Display model results

```python
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2)
```

For four decimal places:

```python
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
```

### K. Common `print()` syntax

```python
print("Hello World")
print("Dataset Shape:", df.shape)
print("Number of Rows:", len(df))
print(f"Accuracy: {accuracy:.2f}")
```

- `print()` displays messages and values.
- Commas separate multiple output items.
- `f"..."` inserts variable values using `{}`.
- `:.2f` formats a number to two decimal places.

### Dataset inspection quick reference

| Requirement | Command |
|---|---|
| First 5 rows | `df.head()` |
| Last 5 rows | `df.tail()` |
| Dataset dimensions | `df.shape` |
| Column names | `df.columns` |
| Data types | `df.dtypes` |
| Dataset information | `df.info()` |
| Statistical summary | `df.describe()` |
| Missing values | `df.isnull().sum()` |
| Unique values | `df['column'].unique()` |
| Value frequencies | `df['column'].value_counts()` |
| Number of rows | `len(df)` |
| Number of columns | `df.shape[1]` |

---

## 5. Data Preprocessing

### A. Handle missing values

Remove rows containing missing values:

```python
df = df.dropna()
```

Fill missing numerical values with the column mean:

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())
```

For rigorous ML workflows, calculate imputation values from training data only, typically through a Scikit-learn pipeline.

### B. Remove a column

```python
df = df.drop('ID', axis=1)
```

- `axis=0`: rows.
- `axis=1`: columns.

### C. Filter rows

```python
df = df[df['Age'] > 18]
```

### D. Remove outliers using a condition

```python
mask = (X.flatten() < 10) & (y < 5)
X, y = X[mask], y[mask]
```

This keeps only observations that satisfy both conditions. Outlier filtering is experiment-dependent and should not be applied automatically.

### E. Encode categorical features

```python
df = pd.get_dummies(df, columns=['Category'], dtype=int)
```

This converts categorical values into indicator columns. Fit encoding steps on training data only when they are part of a predictive pipeline.

---

## 6. Selecting Features and Targets

### Multiple input features

```python
X = df.drop('target', axis=1)
y = df['target']
```

### Single input feature

```python
X = df[['AveRooms']].values
y = df['MedHouseVal'].values
```

### Select columns by position

```python
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

This assumes the last column is the target.

**Remember:**

- `df['column']` returns a Series.
- `df[['column']]` returns a DataFrame.
- `X` is generally 2D.
- `y` is generally 1D.

---

## 7. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

| Parameter | Meaning |
|---|---|
| `test_size=0.2` | 20% test data |
| `test_size=0.3` | 30% test data |
| `random_state=42` | Reproducible split |
| `shuffle=True` | Shuffle data before splitting |
| `stratify=y` | Preserve class proportions for suitable classification tasks |

Classification example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)
```

---

## 8. Feature Scaling

### StandardScaler

Standardizes features to approximately zero mean and unit variance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### MinMaxScaler

Scales features to a specified range, usually 0 to 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

> **Avoid data leakage:** fit the scaler on training data, then use that same scaler to transform the test data.

- `fit_transform()` learns the scaling parameters and transforms the training data.
- `transform()` applies those learned parameters to new data.

Scaling is especially useful for SVM, KNN, neural networks, and gradient descent. Tree-based models generally do not require it.

---

## 9. Common Model Workflow

Most supervised Scikit-learn models follow this pattern:

```python
model = SomeModel()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

| Function | Purpose |
|---|---|
| `SomeModel()` | Create model |
| `.fit()` | Train model |
| `.predict()` | Generate predictions |
| `.fit_predict()` | Fit and return labels, commonly for clustering |
| `.score()` | Return a model-specific score |
| `.get_params()` | Display model parameters |

---

## 10. Regression Experiments

### A. Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### B. Polynomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_poly, y_train)

pred = model.predict(poly.transform(X_test))
```

Use the same fitted transformer on the test data.

### C. Ridge Regression

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Ridge uses L2 regularization.

### D. Lasso Regression

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Lasso uses L1 regularization and can reduce some coefficients to zero.

### E. Regression evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
```

- **MSE:** average squared prediction error.
- **RMSE:** prediction error in the target's units.
- **R²:** coefficient of determination; test-set values can be negative.

### F. Manual Gradient Descent shortcuts

```python
np.mean(error)
np.mean(error * X.flatten())
```

These averages are commonly used to calculate gradients for one-feature linear regression.

### G. Normal Equation

```python
X_b = np.c_[np.ones(len(X)), X]
theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
```

This computes regression coefficients directly using the pseudoinverse rather than iterative gradient descent.

---

## 11. Classification Experiments

### A. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Common kernels: `linear`, `poly`, `rbf`.

Scaling is usually important for SVM.

### B. Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='entropy', random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

- `entropy`: uses entropy/information gain.
- `gini`: uses Gini impurity.

### C. Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Common variants:

- `GaussianNB`: continuous features modeled with Gaussian distributions.
- `MultinomialNB`: non-negative count or frequency features.
- `BernoulliNB`: binary-valued features.

### D. Neural Networks / MLP

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
```

- `(100,)`: one hidden layer with 100 neurons.
- `(50, 25)`: two hidden layers with 50 and 25 neurons.
- `max_iter`: maximum training iterations.

Feature scaling is generally recommended.

### E. Classification evaluation

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

- **Accuracy:** fraction of correct predictions.
- **Precision:** fraction of predicted positives that are correct.
- **Recall:** fraction of actual positives detected.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** compares actual and predicted classes.

---

## 12. Clustering Experiments

Clustering is generally unsupervised and groups observations without using target labels during training.

### A. K-Means

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3, random_state=42, n_init=10
)
labels = model.fit_predict(X)
```

Useful attributes:

```python
model.cluster_centers_
model.inertia_
```

- `cluster_centers_`: cluster centroids.
- `inertia_`: within-cluster sum of squared distances.
- `n_clusters`: requested number of clusters.

### B. Elbow Method

```python
wcss = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k, random_state=42, n_init=10
    )
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()
```

Choose a reasonable elbow point where further increases in K provide diminishing improvements. The elbow is not always perfectly clear.

### C. Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)
```

Agglomerative clustering starts with individual observations and progressively merges groups according to a linkage rule.

### D. Silhouette Score

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(score)
```

A higher silhouette score often indicates better-defined clusters. The score generally ranges from -1 to 1 and requires valid cluster labels with at least two clusters.

---

## 13. Evaluation Metrics

### Regression

```python
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
```

| Metric | Meaning | Preferred direction |
|---|---|---|
| MSE | Mean squared error | Lower |
| RMSE | Root mean squared error | Lower |
| R² | Fit relative to a baseline | Generally higher |

### Classification

```python
accuracy_score(y_test, pred)
confusion_matrix(y_test, pred)
classification_report(y_test, pred)
```

| Metric | Meaning |
|---|---|
| Accuracy | Correct predictions divided by total predictions |
| Precision | Correct positive predictions divided by all predicted positives |
| Recall | Correct positive predictions divided by actual positives |
| F1-score | Harmonic mean of precision and recall |
| Confusion matrix | Summary of actual versus predicted classes |

For imbalanced datasets, accuracy alone may be misleading.

### Clustering

Use metrics such as inertia and silhouette score. These assess cluster structure rather than supervised prediction accuracy.

---

## 14. Common Graph Shortcuts 📊

### A. Create a figure

```python
plt.figure(figsize=(8, 4))
```

`figsize=(8, 4)` sets the width and height in inches.

### B. Scatter plot

Used to display the relationship between two numerical variables.

```python
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.show()
```

Customize transparency and point size:

```python
plt.scatter(X, y, alpha=0.4, s=10)
```

### C. Line plot

Used for trends, fitted lines, cost convergence, and metric comparisons.

```python
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot')
plt.show()
```

Customize the line:

```python
plt.plot(x, y, color='red', linewidth=2, label='Model')
plt.legend()
```

### D. Histogram

Used to inspect the distribution of a numerical variable.

```python
plt.hist(data, bins=20, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution')
plt.show()
```

`bins` controls the number of intervals.

### E. Plot a fitted regression line

```python
xs = np.linspace(X.min(), X.max(), 100)
ys = t0 + t1 * xs

plt.scatter(X.flatten(), y)
plt.plot(xs, ys, color='red')
plt.show()
```

Here, `t0` is the intercept and `t1` is the coefficient for a single feature.

### F. Plot cost convergence

```python
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.show()
```

This shows how the cost changes over optimization iterations.

### G. Multiple plots in one figure

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, y)
axes[0].set_title('Line Plot')

axes[1].scatter(x, y)
axes[1].set_title('Scatter Plot')

plt.tight_layout()
plt.show()
```

- `1, 2`: one row and two columns.
- `axes[0]`, `axes[1]`: individual subplot axes.
- `tight_layout()`: improves spacing.

For two plots vertically:

```python
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
```

### H. Add labels, title, legend, and grid

```python
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Model Results')
plt.legend()
plt.grid(True)
plt.show()
```

Use `legend()` when plotted objects have labels.

### I. Bar chart

```python
plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
```

Useful for comparing category counts or model scores.

### J. Pie chart

```python
plt.pie(values, labels=categories, autopct='%1.1f%%')
plt.show()
```

Useful for showing proportions.

### K. Confusion matrix plot

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.show()
```

### L. Save a graph

```python
plt.savefig('graph.png', dpi=300, bbox_inches='tight')
plt.show()
```

Call `savefig()` before `show()` to reliably save the intended figure.

### Graph shortcut table

| Requirement | Function |
|---|---|
| Relationship between two variables | `plt.scatter()` |
| Trend or fitted line | `plt.plot()` |
| Numerical distribution | `plt.hist()` |
| Category comparison | `plt.bar()` |
| Multiple plots | `plt.subplots()` |
| Axis labels | `plt.xlabel()`, `plt.ylabel()` |
| Graph title | `plt.title()` |
| Legend | `plt.legend()` |
| Grid | `plt.grid(True)` |
| Improve plot spacing | `plt.tight_layout()` |
| Save graph | `plt.savefig()` |
| Display graph | `plt.show()` |

---

## 15. Useful Python and NumPy Shortcuts

| Syntax | Meaning |
|---|---|
| `len(X)` | Number of samples |
| `X.shape` | Array dimensions |
| `X.flatten()` | Convert an array to 1D |
| `X.T` | Matrix transpose |
| `np.mean(error)` | Average error |
| `np.sum(error)` | Sum of errors |
| `np.sqrt(mse)` | Square root of MSE |
| `np.ones(len(X))` | Array of ones |
| `np.zeros(len(X))` | Array of zeros |
| `np.c_[a, b]` | Join arrays as columns |
| `np.linspace(a, b, 100)` | Generate 100 evenly spaced values |
| `np.arange(2, 11)` | Integers from 2 through 10 |
| `range(2, 11)` | Iterate through integers 2 through 10 |
| `np.linalg.pinv(A)` | Matrix pseudoinverse |
| `df.drop('target', axis=1)` | Remove a column |
| `df['column'].values` | Convert a column to NumPy values |
| `plt.tight_layout()` | Improve graph spacing |

---

## 16. Hyperparameter Tuning

Hyperparameters are model settings selected outside ordinary parameter learning.

| Model | Hyperparameter | Purpose |
|---|---|---|
| Polynomial Regression | `degree` | Polynomial complexity |
| Ridge / Lasso | `alpha` | Regularization strength |
| SVM | `C` | Regularization and margin trade-off |
| SVM with RBF kernel | `gamma` | Influence of training points |
| K-Means | `n_clusters` | Number of clusters |
| Decision Tree | `max_depth` | Maximum tree depth |
| MLP | `hidden_layer_sizes` | Hidden-layer architecture |
| Gradient Descent | `lr` | Learning rate |
| Gradient Descent | `epochs` | Number of iterations |

### GridSearchCV example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

model = GridSearchCV(
    SVC(),
    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    cv=5
)

model.fit(X_train, y_train)

print(model.best_params_)
print(model.best_score_)
```

Cross-validation compares parameter combinations. Keep the test set separate for final evaluation.

---

## 17. Common Mistakes

1. **Incorrect array dimensions:** Most Scikit-learn models expect `X` to be 2D.
2. **Data leakage:** Do not fit scalers or imputers on the test set.
3. **Incorrect polynomial transformation:** Use the same fitted transformer for test features.
4. **Wrong metric:** Choose regression or classification metrics according to the task.
5. **Incorrect target column:** Verify which column the experiment asks you to predict.
6. **Missing random state:** Set a fixed seed when reproducibility is required.
7. **Wrong clustering workflow:** K-Means does not require target labels for fitting.
8. **Unscaled features:** Scaling is important for many distance-based and gradient-based algorithms.
9. **Incorrect file path:** Verify the current working directory and dataset location.
10. **Missing dependencies:** Ensure the required libraries and files are available.
11. **Incorrect plot order:** Generate evenly spaced `xs` values for a smooth fitted regression line.
12. **Overfitting:** A model can perform well on training data but poorly on unseen data.
13. **Ignoring class imbalance:** Accuracy can hide poor minority-class performance.
14. **Misreading the elbow graph:** The best K is not necessarily the largest K or the one with the lowest WCSS.
15. **Fitting preprocessing on all data:** Split first, then fit preprocessing steps on the training data.

---

## 18. Quick Viva Reference

| Question | Short answer |
|---|---|
| What is `X`? | Input features |
| What is `y`? | Target variable |
| Why split the dataset? | To evaluate generalization on unseen data |
| What does `fit()` do? | Trains the model |
| What does `predict()` do? | Generates predictions |
| Why use `StandardScaler`? | To standardize feature scales |
| What does `df.head()` do? | Displays the first five rows |
| What does `df.describe()` do? | Displays summary statistics |
| What does `df.info()` do? | Displays column data types and non-null counts |
| What does `df.shape` return? | Number of rows and columns |
| What is overfitting? | Learning training-specific patterns that generalize poorly |
| What is underfitting? | Failing to capture important patterns |
| What is regularization? | Discourages overly complex models |
| What does MSE measure? | Average squared prediction error |
| What does R² measure? | Model fit relative to a baseline |
| What is Gradient Descent? | Iteratively updates parameters to reduce a cost function |
| What is the Normal Equation? | Direct matrix-based computation of linear regression coefficients |
| What is K-Means? | A clustering algorithm based on centroids |
| What is the elbow method? | A method for choosing K by examining WCSS |
| Why use a confusion matrix? | To summarize actual versus predicted classes |
| Why use cross-validation? | To evaluate models across multiple data splits |
| What is a hyperparameter? | A model setting selected outside ordinary parameter learning |

---

## 🎯 Final Exam Memory Trick

Remember this universal sequence:

**Load → Explore → Select → Split → Scale → Fit → Predict → Evaluate → Plot**

For every experiment, identify four things first:

1. **Dataset:** Where does the data come from?
2. **Algorithm:** Which model or mathematical method is required?
3. **Evaluation:** Which metrics does the experiment ask for?
4. **Visualization:** Which graphs must be displayed?

Learn the common workflow first, then memorize the functions unique to each algorithm. This is much more efficient than memorizing every program line by line.

Happy learning, and good luck with your ML lab! 🚀
