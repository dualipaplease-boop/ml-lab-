import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Load data
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False , parser="liac-arff")
X = data.data / 255.0
y = data.target.astype(int)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=2000, random_state=42, stratify=y)

# 3. Preprocess: not needed (already scaled to 0-1)

# 4. Train + 5. Evaluate
results = []
for k in [1, 3, 5, 11]:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train, y_train)

    start = time.time()
    y_pred = model.predict(X_test)
    t = time.time() - start

    results.append([k, accuracy_score(y_test, y_pred),
                    f1_score(y_test, y_pred, average='macro'), t])

df = pd.DataFrame(results, columns=['K', 'Accuracy', 'F1', 'Time (s)'])
print(df)

plt.plot(df['K'], df['Accuracy'], marker='o')
plt.xlabel('K'); plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs K')
plt.show()