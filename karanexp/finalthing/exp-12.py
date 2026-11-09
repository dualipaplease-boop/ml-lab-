import time
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load data
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False , parser="liac-arff")
X = data.data / 255.0
y = data.target.astype(int)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=4800, test_size=1200, random_state=42, stratify=y)

# 3. Preprocess: scale, then PCA (784 -> 50 features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=50, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 4. Train + 5. Evaluate
models = {
    'Linear': SVC(kernel='linear', C=1.0),
    'Polynomial': SVC(kernel='poly', degree=3, C=1.0, gamma='scale'),
    'RBF': SVC(kernel='rbf', C=1.0, gamma='scale'),
}

results = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    t = time.time() - start

    y_pred = model.predict(X_test)
    results.append([name,
                    accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, average='macro'),
                    recall_score(y_test, y_pred, average='macro'),
                    f1_score(y_test, y_pred, average='macro'), t])

df = pd.DataFrame(results, columns=['Kernel', 'Accuracy', 'Precision',
                                    'Recall', 'F1', 'Time (s)'])
print(df)