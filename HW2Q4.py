import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os

def clean_text(text):
    """Converts text to lowercase and strips non-alphabetic characters from the start and end."""
    text = text.lower().strip()
    start, end = 0, len(text) - 1
    while start <= end and not text[start].isalpha():
        start += 1
    while end >= start and not text[end].isalpha():
        end -= 1
    return text[start:end+1]

# Set the directory path to where you've extracted the dataset
directory = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW2\sentenceclassification\SentenceCorpus\labeled_articles'

# Initialize dictionaries to hold categories and vocabulary
categories = {}
vocabulary = {}
X = []
y = []

print("Processing text files to build vocabulary and categories...")
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        with open(os.path.join(directory, filename), 'r') as file:
            for line in file:
                line = line.strip()
                if "###" in line or "--" not in line:
                    continue  # Skip headers and irrelevant lines
                label, sentence = line.split("--")
                words = [clean_text(word) for word in sentence.split()]
                if label not in categories:
                    categories[label] = len(categories)
                y.append(categories[label])
                for word in words:
                    if word not in vocabulary:
                        vocabulary[word] = len(vocabulary)
                X.append([vocabulary[word] for word in words])

# Converting X to a "bag of words" matrix
X_bow = np.zeros((len(X), len(vocabulary)))
for i, doc in enumerate(X):
    for word_idx in doc:
        X_bow[i, word_idx] += 1

# Defining alpha values to test
alpha_values = np.logspace(-15, 5, num=21, base=2)
accuracy_results = []

print("Training and evaluating MultinomialNB models with varying alpha...")
for alpha in alpha_values:
    accuracies = []
    for seed in range(10):  # Repeat experiment 10 times with different random seeds
        X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=seed)
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    avg_accuracy = np.mean(accuracies)
    std_dev = np.std(accuracies)
    accuracy_results.append((alpha, avg_accuracy, std_dev))

    # Print the results for the current alpha
    print(f"Alpha: {alpha:.2e}, Average Accuracy: {avg_accuracy:.4f}, Standard Deviation: {std_dev:.4f}")

# Extracting best alpha, accuracies, and standard deviations for plotting
alphas, avg_accuracies, std_devs = zip(*accuracy_results)

# Plotting average accuracy against alpha
plt.errorbar(alphas, avg_accuracies, yerr=std_devs, fmt='-o', capsize=5)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Average Accuracy')
plt.title('MultinomialNB Accuracy vs Alpha')
plt.legend(['Average Accuracy ± STD'])
plt.show()

# Finding the best alpha
best_alpha = alphas[np.argmax(avg_accuracies)]
print(f"Best alpha for maximizing average accuracy: {best_alpha:.2e}")
print(f"Best average accuracy: {np.max(avg_accuracies):.4f}")

# Retraining with the best alpha
model = MultinomialNB(alpha=best_alpha)
model.fit(X_bow, y)
# Extracting the top 5 words for each class
feature_names = list(vocabulary.keys())
for i, class_label in enumerate(model.classes_):
    top5_indices = model.feature_log_prob_[i].argsort()[-5:]
    top5_words = [feature_names[index] for index in top5_indices]
    print(f"Class {class_label}: Top 5 words: {', '.join(top5_words)}")