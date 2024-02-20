import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import statistics as stat

# Paths to the datasets
X_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW2\tests\X_snts.csv'  # TODO: Replace with your path to X_snts.csv
y_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW2\tests\y_snts.csv'  # TODO: Replace with your path to y_snts.csv
word_dict_path = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW2\tests\textmapper.csv'  # TODO: Replace with your path to word_mapping.csv

# Loading the datasets
X = np.loadtxt(X_path, delimiter=',')
y = np.loadtxt(y_path, delimiter=',')

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the range of alpha values to test
alpha_val = np.linspace(2**-15, 2**5, 1000)
acc = []

# Training and evaluating the model for each alpha value
for a in alpha_val:
    classifier = MultinomialNB(alpha=a)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))

# Finding the alpha value that yields the maximum accuracy
max_acc_index = np.argmax(acc)
max_acc = acc[max_acc_index]
optimal_alpha = alpha_val[max_acc_index]

# Evaluating the classifier with the optimal alpha across different random states
random_state = np.linspace(0, 100, 10)
acc_vals = [accuracy_score(y_test, MultinomialNB(alpha=optimal_alpha).fit(X_train, y_train).predict(X_test))
            for state_val in random_state for X_train, X_test, y_train, y_test in [train_test_split(X, y, test_size=0.2, random_state=int(state_val))]]

# Calculating the mean and standard deviation of the accuracies
mean_accuracy = stat.mean(acc_vals)
sd_accuracy = stat.stdev(acc_vals)

# Re-evaluating the model across a new range of alpha values to plot accuracy
alpha_val2 = np.linspace(2**-15, 2**5, 100)
meanAccVals, stdevVals = [], []


for a_index, a in enumerate(alpha_val2):
    acc_vals = []
    for state_index, state_val in enumerate(random_state):
        # Split the data for this random state
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(state_val))
        # Train the classifier
        classifier = MultinomialNB(alpha=a)
        classifier.fit(X_train, y_train)
        # Make predictions and calculate accuracy
        y_pred = classifier.predict(X_test)
        acc_vals.append(accuracy_score(y_test, y_pred))
    meanAccVals.append(stat.mean(acc_vals))
    stdevVals.append(stat.stdev(acc_vals))
    print(f"Alpha {a}: Mean Accuracy = {meanAccVals[-1]}, Std Dev = {stdevVals[-1]}")

# Plotting the accuracy against alpha values with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(alpha_val2, meanAccVals, label='Average Accuracy')
plt.fill_between(alpha_val2, [m + s for m, s in zip(meanAccVals, stdevVals)], [m - s for m, s in zip(meanAccVals, stdevVals)], color='lightgray', alpha=0.5, label='±1 std. dev.')
plt.axvline(x=alpha_val2[np.argmax(meanAccVals)], color='red', linestyle='--', label='Optimal Alpha')
plt.xscale('log')
plt.title('Average Accuracy vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Finding the net optimal alpha and its corresponding average accuracy
net_optimal_alpha = alpha_val2[np.argmax(meanAccVals)]
max_avg_acc = max(meanAccVals)

# Training the classifier with the net optimal alpha and obtain feature importances
classifier = MultinomialNB(alpha=net_optimal_alpha).fit(X_train, y_train)

# Mapping indices back to words using a dictionary from the CSV file
word_dict = pd.read_csv(word_dict_path, header=None, index_col=0)[1].to_dict()


# Class labels
labels = {0: 'MISC', 1: 'AIMX', 2: 'OWNX', 3: 'CONT', 4: 'BASE'}

# Getting the indices of the top features for each class
top_indices = np.argsort(classifier.feature_log_prob_, axis=1)[:, ::-1][:, :5]

# Mapping indices to words and print them
for i, class_label in enumerate(labels.values()):
    top_words = [word_dict[index] for index in top_indices[i]]
    print(f"{class_label}: Top 5 words: {', '.join(top_words)}")

# Outputs
print(f"Max Average Accuracy: {max_avg_acc}")
print(f"Optimal Alpha: {net_optimal_alpha}")

# Printing a summary of findings
print("\nSummary of Model Performance:")
print(f"Max Accuracy (Single Split): {max_acc:.4f} at Alpha: {optimal_alpha:.4e}")
print(f"Max Average Accuracy (Multiple Splits): {max_avg_acc:.4f} at Alpha: {net_optimal_alpha:.4e}")
print(f"Mean Accuracy (Multiple Splits): {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy (Multiple Splits): {sd_accuracy:.4f}\n")

# Dictionary to map indices to words for feature importance
word_dict = pd.read_csv(word_dict_path, header=None, index_col=0)[1].to_dict()

# Extracting the class labels from the model
class_labels = {index: label for index, label in enumerate(classifier.classes_)}

# Printing the top 5 influential words for each class
print("Top 5 Influential Words per Class based on Feature Log Probabilities:")
for index, class_label in class_labels.items():
    top_indices = classifier.feature_log_prob_[index].argsort()[-5:][::-1]  # Get indices of top 5 features
    top_words = [word_dict[word_index] for word_index in top_indices]
    print(f"Class {class_label} ({labels[class_label]}): {', '.join(top_words)}")
