
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=cols)
X, y = df.drop('Outcome', axis=1), df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Without scaling
m1 = LogisticRegression(max_iter=1000, random_state=42)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)

# With scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m2 = LogisticRegression(max_iter=1000, random_state=42)
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)

# Compare metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores1 = [accuracy_score(y_test, p1), precision_score(y_test, p1),
           recall_score(y_test, p1), f1_score(y_test, p1)]
scores2 = [accuracy_score(y_test, p2), precision_score(y_test, p2),
           recall_score(y_test, p2), f1_score(y_test, p2)]

print(pd.DataFrame({'Metric': metrics, 'Without Scaling': scores1,
                    'With Scaling': scores2}).to_string(index=False))

# Plot comparison
plt.plot(metrics, scores1, marker='o', label='Without Scaling')
plt.plot(metrics, scores2, marker='o', label='With Scaling')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('Logistic Regression Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
