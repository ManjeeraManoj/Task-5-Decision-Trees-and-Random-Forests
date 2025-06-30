# Task 5: Random Forests and Decision Trees

Using the Heart Disease dataset, train and assess Random Forest and Decision Tree classifiers. Study overfitting, pruning, feature importance, tree-based models, and model comparison.

## Actions Taken:

### 1. **Preparation of Data**
The `heart.csv` dataset was loaded, and any missing values were eliminated.
- Divide the target variable (`target`) from the features (`X`).
The 80/20 train-test split was used.

### 2. The Decision Tree Classifier
- Used `sklearn.tree.DecisionTreeClassifier` to train a simple decision tree.
The decision tree was visualized and saved as `decision_tree.png`.
After noticing overfitting, a **pruned tree** was trained using `max_depth=4`.

### 3. The Random Forest Classifier
100 trees were used to train a `RandomForestClassifier`.
Its performance was contrasted with that of the single decision tree.
**Feature importances** were computed and plotted as `feature_importance.png`.

### 4. **Metrics of Evaluation**
All models' **Accuracy** was measured.
The **Classification Report** (precision, recall, and F1-score) was printed.



## 📊 Results:

| Model               | Accuracy |
|--------------------|----------|
| Decision Tree       | ~0.80 (example) |
| Pruned Tree (depth=4) | ~0.78 |
| Random Forest       | ~0.85 |
| Cross-Validation (RF) | ~0.83 |


## 🛠 Tools & Libraries Used:

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
