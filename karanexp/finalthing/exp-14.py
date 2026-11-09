import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
data = fetch_openml('mnist_784', version=1, as_frame=False,parser="liac-arff")
X = data.data / 255.0
y = data.target.astype(int)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

# 3. Preprocess: not needed (already scaled to 0-1)

# 4. Train + 5. Evaluate
losses, accuracies = {}, {}
for act in ['logistic', 'tanh', 'relu']:      # logistic = sigmoid
    model = MLPClassifier(hidden_layer_sizes=(128, 64),
                          activation=act, max_iter=25, random_state=42)
    model.fit(X_train, y_train)

    losses[act] = model.loss_curve_
    accuracies[act] = accuracy_score(y_test, model.predict(X_test))
    print(act, accuracies[act])

for act in losses:
    plt.plot(losses[act], label=act)
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.title('Loss Convergence')
plt.legend()
plt.show()