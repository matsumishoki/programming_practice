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

# 学習率rhoを定義する
rho = 0.5

# 最大のループ回数を定義する
max_iteration = 100

# 重みベクトルを定義する
w = np.random.randn(64)

# 外側のループ
for i in range(max_iteration):
    print i

    # 内側のループ
    for (x_i, t_i) in zip(X, T):
        g_i = np.inner(w, X)
        # wを修正する方法の条件分岐

    # 予測クラスと正解クラスラベルの真値を比較する

# wを可視化する
