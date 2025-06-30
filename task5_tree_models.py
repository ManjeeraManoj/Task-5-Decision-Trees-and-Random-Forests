import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load dataset
df = pd.read_csv("heart.csv")

# Print column names to confirm target
print("Columns:", df.columns.tolist())

# Drop rows with missing values
df.dropna(inplace=True)

# Set target column (usually 'target' in heart datasets)
target = 'target'
X = df.drop(columns=target)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree (default)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# Plot tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png")
plt.close()

# Evaluate Decision Tree
tree_acc = accuracy_score(y_test, tree_pred)
print(f"Decision Tree Accuracy: {tree_acc:.2f}")

# Pruned Decision Tree
pruned_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
pruned_tree.fit(X_train, y_train)
pruned_pred = pruned_tree.predict(X_test)
pruned_acc = accuracy_score(y_test, pruned_pred)
print(f"Pruned Tree Accuracy (max_depth=4): {pruned_acc:.2f}")

# Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
forest_acc = accuracy_score(y_test, forest_pred)
print(f"Random Forest Accuracy: {forest_acc:.2f}")

# Feature Importance
importances = pd.Series(forest.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Cross-validation
cv_scores = cross_val_score(forest, X, y, cv=5)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")

# Classification report for Random Forest
print("\nClassification Report (Random Forest):\n", classification_report(y_test, forest_pred))
