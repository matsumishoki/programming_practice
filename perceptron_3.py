# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:08:48 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# データセットを読み込む
digits = load_digits(2)
X = digits.data
T = digits.target
num_examples = len(X)
for i in range(num_examples):
    if T[i] == 0:
        T[i] = -1

# 学習率rhoを定義する
rho = 0.5

# 最大のループ回数を定義する
max_iteration = 100

# 重みベクトルを定義する
w = np.random.randn(64)

# 外側のループ
for i in range(max_iteration):

    # 内側のループ
    for (x_i, t_i) in zip(X, T):
        g_i = np.inner(w, x_i)
        # wを修正する方法の条件分岐
        if g_i * t_i < 0:
            t_i = np.sign(t_i)
            w_new = w + rho * t_i * x_i
        else:
            w_new = w
        w = w_new
    # 予測クラスと正解クラスラベルの真値を比較する
    y = np.sign(np.inner(w, X))
    num_correct = np.sum(y == T)
    correct_accuracy = num_correct / float(num_examples) * 100
    print correct_accuracy
    if correct_accuracy == 100.0:
        break
# wを可視化する
