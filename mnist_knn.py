# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:07:43 2018

@author: matsumi
"""
import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap
from pandas import Series
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

#    print ("x_train.shape:", x_train.shape)
#    print ("t_train.shape:", t_train.shape)

    # 60000ある訓練データセットを54000と6000の評価のデータセットに分割する
    x_train, x_valid, t_train, t_valid = train_test_split(
        x_train, t_train, test_size=0.1, random_state=100)

    print ("x_train.shape:", x_train.shape)
    print ("t_train.shape:", t_train.shape)
    print ("x_valid.shape:", x_valid.shape)
    print ("t_valid.shape:", t_valid.shape)
    print ("x_test.shape:", x_test.shape)
    print ("t_test.shape:", t_test.shape)

    num_train = len(x_train)
    num_valid = len(x_valid)
    num_test = len(x_test)

    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元
    
    # Knn
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(x_train, t_train)
    predict = knn.predict(x_valid)
    accuracy = metrics.accuracy_score(t_valid, predict)
    print("accuracy", accuracy)
#    accuracy = []
#    for k in range(1,2):
#        knn = KNeighborsClassifier(n_neighbors=k)
#        knn.fit(x_train, t_train)
#        predict = knn.predict(x_valid)
#        accuracy.append(metrics.accuracy_score(t_valid, predict))
#        print("accuracy:", accuracy)
#        plt.plot(accuracy)
