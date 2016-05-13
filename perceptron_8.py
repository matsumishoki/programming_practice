# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:19:14 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


class Perceptron(object):
    def __init__(self, learning_rate=0.5, max_iteration=100):
        self.learningrate = learning_rate
        self.max_iteration = max_iteration

    def fit(self, X, y):
        y = self._convert_label(y)
        dimension = X.shape[1]
        w = np.random.randn(dimension)
        for i in range(self.max_iteration):
                for (x_i, y_i) in zip(X, y):
                    g_i = np.inner(w, x_i)
                    if g_i * y_i < 0:
                        w = w + self.learningrate * y_i * x_i
                    self.w = w
        y = self._turn_back_label(y)
        return self

    def predict(self, X):
        y = np.sign(np.inner(self.w, X))
        return self._turn_back_label(y)

    def score(self, X, y):
        predict_y = self.predict(X)
        correct_rate = np.mean(predict_y == y)
        return correct_rate

    def _convert_label(self, y):
        y = y.copy()
        y[y == 0] = -1
        return y

    def _turn_back_label(self, y):
        y = y.copy()
        y[y == -1] = 0
        return y

    def finish_learning_w(self, X, y):
        if self.score(X, y) == 1.0:
            return self.w

if __name__ == '__main__':
    digits = load_digits(2)
    X = digits.data
    t = digits.target

    classfier = Perceptron()
    classfier.fit(X, t)
    y = classfier.predict(X)
    print "predict_y:", y
    print "t", t
    score = classfier.score(X, t)
    print "score", score
    w = classfier.finish_learning_w(X, t)
    print "w:", w
    plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
