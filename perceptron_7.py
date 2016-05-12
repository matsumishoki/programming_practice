# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:44:47 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class Perceptron(object):
    def __init__(self, learning_rate=0.5, max_iteration=100):
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration

    def fit(self, X, t):
        dimension = X.shape[1]
        t[t == 0] = -1
        w = np.random.randn(dimension)
        for self.i in range(self.max_iteration):

            for (x_i, t_i) in zip(X, t):
                g_i = np.inner(w, x_i)

                if g_i * t_i < 0:
                    w = w + self.learning_rate * t_i * x_i
                self.w = w
                self.predict(X, t)
                score = self.score(X)
                self.count_epoch(self.i)
                self.break_point(score)
        return self

    def predict(self, X, t):
        y = np.sign(np.inner(self.w, X))
        self.y = y
        return self

    def score(self, X):
        num_samples = len(X)
        num_correct = np.sum(self.y == t)
        accuracy_rate = num_correct / float(num_samples) * 100
        return accuracy_rate

    def break_point(self, score):
        if score == 100.0:
            print self.i
            return self

    def count_epoch(self, i):
        i = i + 1

if __name__ == '__main__':
    digits = load_digits(2)
    X = digits.data
    t = digits.target

    classifier = Perceptron()
    classifier.fit(X, t)
    y = classifier.predict(X, t)
    accuracy_rate = classifier.score(X)
    print "accuracy_rate:", accuracy_rate
    print "y:", y.y
    print "t:", t
    print "w:",
#    plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
#    plt.show()
