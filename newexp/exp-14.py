# newexp/exp-14.py
# Optimized: MLP on MNIST Activations → fetch_openml + simplified logging
# Original: 42 lines → Optimized: ~35 lines (17% reduction)
# Code reduction: fetch_openml replaces URL-based loading + simplified activation name mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("Dataset processing complete.")

# Train with different activations
history_loss = {}
test_accuracies = {}
activations = ['logistic', 'tanh', 'relu']

# Simplified activation name mapping (replaced complex ternary conditional)
act_names = {'logistic': 'sigmoid', 'tanh': 'tanh', 'relu': 'relu'}

for act in activations:
    print(f"Training network using activation: {act}")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation=act, max_iter=25, random_state=42)
    mlp.fit(X_train, y_train)
    history_loss[act] = mlp.loss_curve_
    test_accuracies[act] = accuracy_score(y_test, mlp.predict(X_test))
    print(f"Accuracy for {act_names[act]}: {test_accuracies[act]:.4f}")

print(test_accuracies)

# Plot loss convergence
plt.figure(figsize=(10, 5))
for act in activations:
    plt.plot(history_loss[act], label=act_names[act])
plt.title('Loss Convergence Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.show()