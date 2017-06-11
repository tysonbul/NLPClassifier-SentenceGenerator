#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk import stem, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import math

from timeit import default_timer as timer

### Globals ###
apply_stemming = False
stopwords = None
plot_outcome_values = False

class MyBayesClassifier():

    def __init__(self, smooth=1):
        self._smooth = smooth # This is for additive smoothing
        self._feat_prob = [] # do not change the name of these vars
        self._feat_counter_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []
        self._classes = [] # Keep track of the class values

    def train(self, X, y):

        def populate_probabilities(reviews, cls):
            len_reviews = reviews.shape[0]
            for index, col_sum in enumerate(np.sum(reviews, axis=0, dtype=float)):
                probability = (col_sum + self._smooth) / (len_reviews + (self._smooth * 2))
                self._feat_prob[cls][index] = probability
                self._feat_counter_prob[cls][index] = 1 - probability
            # Convert to log
            self._feat_prob[cls] = np.log(self._feat_prob[cls])
            self._feat_counter_prob[cls] = np.log(self._feat_counter_prob[cls])


        self._classes = np.unique(y)
        Ncls, Nfeat = len(self._classes), X.shape[1]
        self._Ncls, self._Nfeat = Ncls, Nfeat
        self._feat_prob = np.zeros((Ncls, Nfeat))
        self._feat_counter_prob = np.zeros((Ncls, Nfeat))
        self._class_prob = np.zeros(Ncls)

        num_reviews = len(y)
        reviews_by_class = []

        # Separate reviews by each type of class and populate probabilities
        for index, val in enumerate(self._classes):
            # Append reviews from X on indexes from y where y == class value
            reviews_by_class.append(X[tuple(np.where(y==val))])
            # Calculate class probability while we're here
            self._class_prob[index] = math.log(float(len(reviews_by_class[index])) / float(num_reviews))
            # Populate probabilities given the class
            populate_probabilities(reviews_by_class[index], index)

    def predict(self, X):

        pred = np.zeros(len(X))
        for index, document in enumerate(X):
            counter_document = 1-document
            vals = []
            # Calculate each classes probability and take the max
            for cls, val in enumerate(self._classes):
                numerator = np.dot(document, self._feat_prob[cls]) + np.dot(counter_document, self._feat_counter_prob[cls])
                numerator += self._class_prob[cls]
                vals.append((cls, numerator))
            # Assign class to prediction for row
            class_with_max_prob = max(vals, key=lambda v:v[1])
            pred[index] = self._classes[class_with_max_prob[0]]

        return pred

    @property
    def probs(self):
        # please leave this intact, we will use it for marking
        return self._class_prob, self._feat_prob


def clean_data(X):
    """ Clean passed in data
          Checks for:
            - Empty strings
    """
    return [row for row in X if len(row) > 0]

def stem_data(lines):
    """
        Stems each line word by word
        Returns: list of stemmed lines
    """
    stemmer = stem.PorterStemmer()
    stemmed_lines = []
    for line in lines:
        tokens = word_tokenize(line.decode("utf8"))
        stemmed_line = ' '.join([stemmer.stem(token) for token in tokens])
        stemmed_lines.append(stemmed_line)
    return stemmed_lines

def plot_alpha_values(X_train, y_train, X_test, y_test):
    """ Plots output accuracies for varying alpha values [0.1,3.0] """
    x_axis, y_axis = [], []
    for alpha in np.arange(0.1, 3.1, 0.1):
        x_axis.append(alpha)
        clf = MyBayesClassifier(alpha)
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_axis.append(np.mean((y_test - y_pred) == 0))

    plt.plot(x_axis, y_axis)
    plt.ylabel('Accuracy')
    plt.xlabel('Alpha')
    plt.show()


""" 
Here is the calling code

"""

# program_time = timer()

with open('sentiment_data/rt-polarity_utf8.neg', 'r') as f:
    lines_neg = clean_data(f.read().split('\n'))

with open('sentiment_data/rt-polarity_utf8.pos', 'r') as f:
    lines_pos = clean_data(f.read().split('\n'))

data_train = lines_neg[0:5000] + lines_pos[0:5000]
data_test = lines_neg[5000:] + lines_pos[5000:]

if apply_stemming:
    data_train = stem_data(data_train)
    data_test = stem_data(data_test)

y_train = np.append(np.ones((1,5000)), (np.zeros((1,5000))))
y_test = np.append(np.ones((1,331)), np.zeros((1,331)))

# You will be changing the parameters to the CountVectorizer below
vectorizer = CountVectorizer(lowercase=True, stop_words=stopwords,  max_df=1.0, min_df=1, max_features=None,  binary=True)
X_train = vectorizer.fit_transform(data_train).toarray()
X_test = vectorizer.transform(data_test).toarray()
feature_names = vectorizer.get_feature_names()
if not plot_outcome_values:
    clf = MyBayesClassifier(1);
    clf.train(X_train,y_train);
    y_pred = clf.predict(X_test)
    print(np.mean((y_test-y_pred)==0))
else:
    plot_alpha_values(X_train, y_train, X_test, y_test)

# print("Exectution time: {0}".format(timer()-program_time))