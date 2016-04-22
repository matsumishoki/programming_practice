# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:54:08 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# データセットを読み込む
digits = load_digits(2)
X = digits.data
T = digits.target
num_sumples = len(X)

# Tの要素の0を-1にする
for i in range(num_sumples):
    if T[i] == 0:
        T[i] = -1

# wを定義する
w = np.random.randn(64)

# 学習率(rho)を定義する
rho = 0.5

# ループ回数を定義する
max_iteration = 100

# 外側のループ
for i in range(max_iteration):

    # 学習させるループ
    for (x_i, t_i) in zip(X, T):
        g_i = np.inner(w, x_i)
        t_i = np.sign(t_i)

    # wの修正条件
        if t_i * g_i < 0:
            w_new = w + rho * t_i * x_i
        else:
            w_new = w
        w = w_new

    # 予測クラスと正解クラスの値を比較する
    y = np.sign(np.inner(w, X))
    num_correct = np.sum(y == T)
    correct_accuracy = num_correct / float(num_sumples) * 100
    print correct_accuracy
    if correct_accuracy == 100.0:
        break

print 'T:', T
print 'y:', y

# wの可視化
print 'w:', w
plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
