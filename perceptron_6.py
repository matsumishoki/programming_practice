# -*- coding: utf-8 -*-
"""
Created on Mon May 09 14:30:57 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

if __name__ == '__main__':
    digits = load_digits(2)
    X = digits.data
    t = digits.target
    t[t == 0] = -1

    learning_rate = 0.5
    max_iteration = 100
    num_samples = len(X)

    # 重みベクトルを定義する
    w_dimension = 64
    w = np.random.randn(w_dimension)

    for i in range(max_iteration):

        for (x_i, t_i) in zip(X, t):
            g_i = np.inner(w, x_i)

            if t_i * g_i < 0:
                w = w + learning_rate * t_i * x_i

            w = w

        y = np.sign(np.inner(w, X))
        num_correct = np.sum(y == t)
        correct_rate = num_correct / float(num_samples) * 100
        print "finish_iteration:", i + 1
        print "correct_rate:", correct_rate
        if correct_rate == 100.0:
            break
    print
    print "finish_iteration:", i + 1
    print "correct_rate:", correct_rate
    print "y:", y
    print "t:", t
    print "w:", w
    plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
