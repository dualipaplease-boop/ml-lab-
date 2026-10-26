
import time
import torchvision
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

# Load Fashion-MNIST
train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)

X_train = train.data.numpy().reshape(-1, 784)[:15000] / 255.0
y_train = train.targets.numpy()[:15000]
X_test = test.data.numpy().reshape(-1, 784)[:1500] / 255.0
y_test = test.targets.numpy()[:1500]

# Compare K values
results = []
for k in [1, 3, 5, 11]:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    model.fit(X_train, y_train)

    start = time.time()
    pred = model.predict(X_test)
    elapsed = time.time() - start

    results.append([k, accuracy_score(y_test, pred),
                    f1_score(y_test, pred, average='macro'), elapsed])

df = pd.DataFrame(results, columns=['K', 'Accuracy', 'F1-Score', 'Time'])
print(df.to_string(index=False))

# Plot accuracy and inference time
fig, ax1 = plt.subplots()
ax1.plot(df['K'], df['Accuracy'], marker='o')
ax1.set_xlabel('K Value')
ax1.set_ylabel('Accuracy')

ax2 = ax1.twinx()
ax2.plot(df['K'], df['Time'], marker='s', linestyle='--')
ax2.set_ylabel('Inference Time (s)')

plt.title('KNN: Accuracy vs Efficiency')
plt.tight_layout()
plt.show()
