# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:47:37 2016

@author: matsumi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# データセットを読み込む
digits = load_digits(2)
images = digits.images
T = digits.target
X = digits.data
num_examples = len(X)

# Tの要素の0を-1に書き換える
for i in range(len(T)):
    if T[i] == 0:
        T[i] = -1

print T.shape
print X.shape
print digits.data.shape
plt.matshow(images[0], cmap=plt.cm.gray)
plt.show()

# 学習率を定義する
rho = 0.5

# wを定義する
w = np.random.randn(64)

# 最大のループ回数を定義する
max_iteration = 100

# 一番大きなループ
for i in range(max_iteration):

    # 識別関数の値を計算する
    for (x_i, t_i) in zip(X, T):
        g_i = np.inner(w, x_i)
        # wを更新する
        if t_i * g_i < 0:
            t_i = np.sign(t_i)
            w_new = w + rho * t_i * x_i
    # wの更新が良ければ更新しない
        else:
            w_new = w
        w = w_new

    # 予測クラスと比較する
    y = np.sign(np.inner(w, X))
    num_correct = np.sum(y == T)
    correct_accuracy = num_correct / float(num_examples) * 100
    print 'correct_accuracy:', correct_accuracy
    if correct_accuracy == 100.0:
        print 'w更新',
        print i+1,
        print '回目'
        break

# 正解クラスと予測クラスの値を表示する
print 'y:', y
print 'T:', T

# wを表示する
plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
plt.show()
