import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

# Load dataset
iris = load_iris(as_frame=True)
X = iris.data.drop(columns=['sepal length (cm)', 'sepal width (cm)'])
y = (iris.target == 0).astype(int)

# Plot Iris species
for i in range(3):
    plt.scatter(X.loc[iris.target == i, 'petal length (cm)'],
                X.loc[iris.target == i, 'petal width (cm)'],
                label=iris.target_names[i])

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluate model
print('Accuracy:', accuracy_score(y_test, pred))
print('Precision:', precision_score(y_test, pred))
print('Recall:', recall_score(y_test, pred))
print('F1-score:', f1_score(y_test, pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, pred))

# Plot decision boundary and margins
DecisionBoundaryDisplay.from_estimator(
    model, X_train, response_method='decision_function',
    plot_method='contour', levels=[-1, 0, 1],
    colors='black', linestyles=['--', '-', '--'])

# Plot training data
plt.scatter(X_train[:, 0], X_train[:, 1],
            c=y_train, edgecolors='black', label='Training Data')

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='red',
            label='Support Vectors')

plt.xlabel('Scaled Petal Length')
plt.ylabel('Scaled Petal Width')
plt.title('SVM Decision Boundary and Margins')
plt.legend()
plt.tight_layout()
plt.show()