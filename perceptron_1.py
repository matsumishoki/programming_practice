# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:47:37 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# データセットを読み込む
digits = load_digits()
images = digits.images
T = digits.target
X = digits.data
print T.shape
print X.shape
print digits.data.shape
plt.matshow(images[0], cmap=plt.cm.gray)

# 学習率を定義する
rho = 0.5

# wを定義する
w = np.random.randn(64)

# 最大のループ回数を定義する
max_iteration = 100

# 一番大きなループ
for i in range(max_iteration):
    print i

    # 識別関数の値を計算する
    for (x_i, t_i) in zip(X, T):
        g_i = np.inner(w, x_i)
        # wを更新する
        if t_i * g_i < 0:
            w_new = w + rho * t_i * x_i
            print t_i
    # wの更新が良ければ更新しない
    else:
        w_new = w
    w = w_new

# 予測クラスと比較する

# wを表示する
