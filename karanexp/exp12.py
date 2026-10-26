
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train[:2000].reshape(-1, 784), X_test[:1200].reshape(-1, 784)
y_train, y_test = y_train[:2000], y_test[:1200]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compare SVM kernels
results, predictions = [], {}
for kernel in ['linear', 'poly', 'rbf']:
    model = SVC(kernel=kernel)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    pred = model.predict(X_test)
    predictions[kernel] = pred

    results.append([
        kernel,
        accuracy_score(y_test, pred) * 100,
        precision_score(y_test, pred, average='macro') * 100,
        recall_score(y_test, pred, average='macro') * 100,
        f1_score(y_test, pred, average='macro') * 100,
        elapsed
    ])

df = pd.DataFrame(results, columns=[
    'Kernel', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Time'
])
print(df.round(2).to_string(index=False))

best = df.loc[df['Accuracy'].idxmax(), 'Kernel']
print("Best Kernel:", best)

# Plot confusion matrix
names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cm = confusion_matrix(y_test, predictions[best])

plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(10), names, rotation=45, ha='right')
plt.yticks(range(10), names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - ' + best.upper())

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.show()
