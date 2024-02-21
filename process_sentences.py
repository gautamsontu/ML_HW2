import csv
import os


def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'.
    """
    return s.lower()


def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string.

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle
    of the string should not be removed. E.g. "haven't" should remain unaltered."""

    s = s.strip()
    if len(s) == 0:
        return s
    if not s[0].isalpha():
        return strip_non_alpha(s[1:])
    elif not s[-1].isalpha():
        return strip_non_alpha(s[:-1])
    else:
        return s


def clean(s):
    """ Create a "clean" version of a string
    """
    return to_lower_case(strip_non_alpha(s))


# Directory of text files to be processed

directory = r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW2\sentenceclassification\SentenceCorpus\labeled_articles'

# Learn the vocabulary of words in the corpus
# as well as the categories of labels used per text

categories = {}
vocabulary = {}

num_files = 0
for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    num_files += 1
    print("Processing", filename, "...", end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                if label not in categories:
                    index = len(categories)
                    categories[label] = index

                for word in words:
                    if word not in vocabulary:
                        index = len(vocabulary)
                        vocabulary[word] = index
    print(" done")

n_words = len(vocabulary)
n_cats = len(categories)

print("Read %d files containing %d words and %d categories" % (num_files, len(vocabulary), len(categories)))

print(categories)

# Convert sentences into a "bag of words" representation.
# For example, "to be or not to be" is represented as
# a vector with length equal to the vocabulary size,
# with the value 2 at the indices corresponding to "to" and "be",
# value 1 at the indices corresponding to "or" and "not"
# and zero everywhere else.


X = []
y = []

for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    print("Converting", filename, "...", end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                y.append(categories[label])

                features = n_words * [0]

                bag = {}
                for word in words:
                    if word not in bag:
                        bag[word] = 1
                    else:
                        bag[word] += 1

                for word in bag:
                    features[vocabulary[word]] = bag[word]

                X.append(features)
    print(" done")

# Save X and y to files

with open('X_snts.csv', 'w') as csvfile:
    fw = csv.writer(csvfile, delimiter=',')
    for features in X:
        fw.writerow(features)

with open('y_snts.csv', 'w') as csvfile:
    fw = csv.writer(csvfile, delimiter=',')
    for label in y:
        fw.writerow([label])

print(categories)
csv_file_name = 'textmapper.csv'

with open(csv_file_name, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for key, value in vocabulary.items():
        csv_writer.writerow([value, key])