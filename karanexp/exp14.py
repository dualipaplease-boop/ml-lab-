
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time

# Load and preprocess MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True,
                    as_frame=False, parser='auto')
X, y = X[:30000] / 255.0, y[:30000]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare activation functions
for act in ['logistic', 'relu', 'tanh']:
    model = MLPClassifier(hidden_layer_sizes=(128, 64),
                          activation=act, random_state=42)

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    accuracy = model.score(X_test, y_test) * 100

    print(f'{act}: Time={elapsed:.2f}s, '
          f'Accuracy={accuracy:.2f}%, Iterations={model.n_iter_}')

    plt.plot(model.loss_curve_, label=act)

# Plot convergence
plt.title('Convergence Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
