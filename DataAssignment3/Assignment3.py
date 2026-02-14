
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ================== Load datasets ==================
crime_df = pd.read_csv('crime.csv')
kidney_df = pd.read_csv('kidney_disease.csv')

# ================== Question 1 ==================
print("="*50)
print("Question 1")
print("="*50)

# Extract target column
vc = crime_df['ViolentCrimesPerPop']

# Compute statistics
mean_val = vc.mean()
median_val = vc.median()
std_val = vc.std()
min_val = vc.min()
max_val = vc.max()

print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Standard deviation: {std_val}")
print(f"Minimum: {min_val}")
print(f"Maximum: {max_val}")

# Comments answering the questions:
# 1. Compare mean and median. If mean > median, the distribution is right‑skewed (positive skew);
#    if mean < median, it is left‑skewed (negative skew); if roughly equal, it may be symmetric.
#    Based on the actual output, e.g., if mean > median, there are some large extreme values pulling the mean upward, indicating right skew.
# 2. Extreme values affect the mean more because the mean takes all values into account, while the median only considers the middle position and is robust to outliers.

# ================== Question 2 ==================
print("\n" + "="*50)
print("Question 2")
print("="*50)

# Create histogram
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(vc, bins=30, edgecolor='black')
plt.title('Histogram of Violent Crimes per Pop')
plt.xlabel('ViolentCrimesPerPop')
plt.ylabel('Frequency')

# Create box plot
plt.subplot(1,2,2)
plt.boxplot(vc)
plt.title('Box Plot of Violent Crimes per Pop')
plt.xlabel('ViolentCrimesPerPop')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Comments describing the plots:
# The histogram shows the distribution shape. If there is a long right tail, it indicates some areas with high crime rates, i.e., right skew.
# The line inside the box plot represents the median, and the box spans the interquartile range. Many points outside the box (outliers) suggest extreme values.
# The box plot visually displays the median and the spread of the data. If there are individual points above or below the whiskers, they indicate potential outliers.
# For example, if there are many points beyond the upper whisker, it suggests high‑value outliers.

# ================== Question 3 ==================
print("\n" + "="*50)
print("Question 3")
print("="*50)

# Separate features and labels (adjust column name if needed, e.g., 'Classification')
X = kidney_df.drop('classification', axis=1)  # feature matrix
y = kidney_df['classification']               # label vector

# Split into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Comments explaining:
# 1. If we train and test on the same data, the model may simply memorize the training examples, leading to overfitting and poor performance on unseen data.
# 2. The testing set simulates how the model performs on new, unseen data, allowing us to evaluate its generalization ability. It helps us determine whether the model has truly learned underlying patterns rather than just memorizing.

# ================== Question 4 ==================
print("\n" + "="*50)
print("Question 4")
print("="*50)

# Train KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=0)   # positive class: 0 = CKD
rec = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)

print("Confusion Matrix:")
print(cm)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# Comments explaining:
# True Positive: patients actually having kidney disease and correctly predicted as having it.
# True Negative: patients actually not having kidney disease and correctly predicted as not having it.
# False Positive: patients actually not having kidney disease but incorrectly predicted as having it (misdiagnosis).
# False Negative: patients actually having kidney disease but incorrectly predicted as not having it (missed diagnosis).
# Accuracy alone may be misleading, especially with imbalanced data; e.g., if most samples are negative, predicting all as negative yields high accuracy but misses all positives.
# If missing a kidney disease case is very serious, recall is the most important metric because it measures the model's ability to identify all actual positives. We want to minimize false negatives.

# ================== Question 5 ==================
print("\n" + "="*50)
print("Question 5")
print("="*50)

k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Create results table
results = pd.DataFrame({'k': k_values, 'Accuracy': accuracies})
print(results)

# Find best k
best_k = results.loc[results['Accuracy'].idxmax(), 'k']
print(f"\nHighest test accuracy achieved with k = {best_k}")

# Comments explaining:
# Smaller k makes the model more complex, with a wiggly decision boundary that can capture noise in the training data, leading to overfitting.
# Larger k makes the model simpler, with a smoother decision boundary, which may underfit by missing true patterns in the data.
# Choosing an appropriate k balances bias and variance, improving the model's generalization to unseen data.