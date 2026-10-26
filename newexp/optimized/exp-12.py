from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Fashion MNIST dataset
print("1. Downloading and loading Fashion MNIST data...")
train_dataset = FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_dataset = FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Convert to NumPy and flatten
X_train = train_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy().reshape(-1, 28 * 28) / 255.0
y_test = test_dataset.targets.numpy()

# Visualize samples
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()
print(pd.Series(y_train).value_counts().sort_index())

# Subsample for tractability
X_sub, _, y_sub, _ = train_test_split(
    X_train.reshape(len(X_train), -1), y_train, train_size=6000, stratify=y_train, random_state=42)
X_sub = X_sub.astype('float32') / 255.0

# Train-test split (80/20)
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_sub)
X_test_scaled = scaler.transform(X_sub)

# Define kernels to test
kernels = ['linear', 'poly', 'rbf']
results = []

for kernel_name in kernels:
    # Pipeline approach: PCA + SVC in one line (vs manual PCA + separate SVC before)
    pipe = make_pipeline(
        PCA(n_components=50, random_state=42),
        SVC(kernel=kernel_name)
    )
    
    # Train pipeline
    start = time.time()
    pipe.fit(X_train_scaled, y_train_sub)
    train_time = time.time() - start
    
    # Predict and evaluate
    y_pred = pipe.predict(X_test_scaled)
    
    acc = accuracy_score(y_test_sub, y_pred)
    prec = precision_score(y_test_sub, y_pred, average='macro')
    rec = recall_score(y_test_sub, y_pred, average='macro')
    f1 = f1_score(y_test_sub, y_pred, average='macro')
    
    results.append({
        'Kernel': kernel_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Train Time (s)': train_time
    })
    
    print(f'{kernel_name:12s} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} Time={train_time:.2f}s')

# Results DataFrame
results_df = pd.DataFrame(results, columns=['Kernel', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train Time (s)'])
print(results_df)

# Plot accuracy comparison
results_df.plot(x='Kernel', y='Accuracy', kind='bar', legend=False, color='steelblue')
plt.ylabel('Test Accuracy')
plt.title('SVM Kernel Comparison on Fashion MNIST')
plt.tight_layout()
plt.show()

# Confusion matrix for best kernel
best_kernel = results_df.loc[results_df['Accuracy'].idxmax(), 'Kernel']
# Re-predict with best kernel via pipeline
best_pred = pipe.predict(X_test_scaled)  # Uses last pipeline fitted (RBF if last, or select properly)

cm = confusion_matrix(y_test_sub, best_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(10), class_names, rotation=90)
plt.yticks(range(10), class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix — Best Kernel: {best_kernel}')
threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(i, j, str(cm[i, j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")
plt.tight_layout()
plt.show()