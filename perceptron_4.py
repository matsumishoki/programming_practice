# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:57:45 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits(2)
X = digits.data
T = digits.target
num_sumples = len(X)

# HACK: 1行で書きたい
for i in range(num_sumples):
    if T[i] == 0:
        T[i] = -1

learning_rate = 0.5
w = np.random.randn(64)

max_iteration = 100

for i in range(max_iteration):

    for (x_i, t_i) in zip(X, T):
        g_i = np.inner(w, x_i)

        if g_i * t_i < 0:
            t_i = np.sign(t_i)
            w = w + learning_rate * t_i * x_i

    predict_y = np.sign(np.inner(w, X))
    num_correct = np.sum(predict_y == T)
    correct_adduracy = num_correct / float(num_sumples) * 100
    print "itereation_times:", i + 1
    print correct_adduracy

    if correct_adduracy == 100.0:
        break

print
print "iteration_finish:", i + 1
print "learning_rate:", learning_rate
print "predict_y:", predict_y
print "T:", T

print "w:", w
plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
plt.show
