import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Loading the dataset
file_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW2\mushroom\agaricus-lepiota.data'
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
           'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
           'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
           'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
data = pd.read_csv(file_path, header=None, names=columns)
print("Dataset loaded successfully.")

# Encoding categorical features as integers
label_encoder = LabelEncoder()
for col in data.columns:
    data[col] = label_encoder.fit_transform(data[col])
print("Categorical features encoded.")

# Splitting the data into features and target
X = data.drop('class', axis=1)
y = data['class']
print("Data split into features and target.")

# Splitting the dataset into training set (80%) and test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into training (80%) and test (20%) sets: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

# Defining alpha values to test
alpha_values = np.logspace(-15, 5, num=21, base=2)
performance_metrics = {'alpha': [], 'ROC AUC': [], 'Accuracy': [], 'F1': []}

print("Training CategoricalNB models with varying alpha...")

for alpha in alpha_values:
    # Training CategoricalNB with the current alpha
    model = CategoricalNB(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"Model trained with alpha = {alpha}")

    # Predicting on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Computing metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Metrics for alpha = {alpha}: ROC AUC = {roc_auc:.4f}, Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

    # Storing metrics
    performance_metrics['alpha'].append(alpha)
    performance_metrics['ROC AUC'].append(roc_auc)
    performance_metrics['Accuracy'].append(accuracy)
    performance_metrics['F1'].append(f1)

# Converting metrics to DataFrame for easier handling
performance_df = pd.DataFrame(performance_metrics)

# Plotting
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
ax[0].plot(performance_df['alpha'], performance_df['ROC AUC'], marker='o', linestyle='-', color='b')
ax[0].set_xscale('log')
ax[0].set_xlabel('Alpha')
ax[0].set_ylabel('ROC AUC')
ax[0].set_title('ROC AUC vs Alpha')

ax[1].plot(performance_df['alpha'], performance_df['Accuracy'], marker='o', linestyle='-', color='r')
ax[1].set_xscale('log')
ax[1].set_xlabel('Alpha')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Accuracy vs Alpha')

ax[2].plot(performance_df['alpha'], performance_df['F1'], marker='o', linestyle='-', color='g')
ax[2].set_xscale('log')
ax[2].set_xlabel('Alpha')
ax[2].set_ylabel('F1 Score')
ax[2].set_title('F1 Score vs Alpha')

plt.tight_layout()
plt.show()

# Finding the alpha value that maximizes the AUC
best_alpha = performance_df.loc[performance_df['ROC AUC'].idxmax()]['alpha']
best_performance = performance_df.loc[performance_df['ROC AUC'].idxmax()]
print(f"\nFinding best alpha for maximizing ROC AUC with standard training set: {best_alpha}")
print("Best performance metrics:", best_performance.to_dict())

# Splitting the dataset into training set (1%) and test set (99%)
X_train_small, X_test_large, y_train_small, y_test_large = train_test_split(X, y, test_size=0.99, random_state=42)

# Re-running the experiment with the smaller training set
performance_metrics_small = {'alpha': [], 'ROC AUC': [], 'Accuracy': [], 'F1': []}

for alpha in alpha_values:
    # Training CategoricalNB with the current alpha
    max_categories = max([data[col].nunique() for col in data.columns[1:]])  # Exclude the target column
    model_small = CategoricalNB(alpha=alpha, min_categories=max_categories)
    model_small.fit(X_train_small, y_train_small)

    # Predicting on the larger test set
    y_pred_large = model_small.predict(X_test_large)
    y_pred_proba_large = model_small.predict_proba(X_test_large)[:, 1]

    # Computing metrics
    roc_auc_large = roc_auc_score(y_test_large, y_pred_proba_large)
    accuracy_large = accuracy_score(y_test_large, y_pred_large)
    f1_large = f1_score(y_test_large, y_pred_large)

    # Storing metrics
    performance_metrics_small['alpha'].append(alpha)
    performance_metrics_small['ROC AUC'].append(roc_auc_large)
    performance_metrics_small['Accuracy'].append(accuracy_large)
    performance_metrics_small['F1'].append(f1_large)

# Converting metrics to DataFrame for easier handling
performance_df_small = pd.DataFrame(performance_metrics_small)

# Finding the alpha value that maximizes the AUC for the small training set
best_alpha_small = performance_df_small.loc[performance_df_small['ROC AUC'].idxmax()]['alpha']
best_performance_small = performance_df_small.loc[performance_df_small['ROC AUC'].idxmax()]

print("\nFinding best alpha for maximizing ROC AUC with small training set:")
print(f"Alpha: {best_alpha_small}")
print("Best performance metrics for small training set:", best_performance_small.to_dict())
