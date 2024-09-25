#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    """YOUR CODE HERE FOR Q1. Also modify knn.py to implement KNN predict."""
    k_values = [1, 3, 10]
    for k in k_values:
        model = KNN(k)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("k: %d" % k)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)
        if k == 1:
            plot_classifier(model, X, y)
            fname = Path("..", "figs", "q1_3_citiesSmallKNNPlot.pdf")
            plt.savefig(fname)
            valModel = KNeighborsClassifier(1)
            valModel.fit(X,y)
            plot_classifier(valModel, X, y)
            fname = Path("..", "figs", "q1_3_citiesSmallSKLearnKNNPlot.pdf")
            plt.savefig(fname)
    # 1.4 each training point is its own neighbour, so error is zero. The model just memorizes the training data so when you pass training data in to calculate training error, it just matches the given point to the point it memorizes
    # 1.5 to find the best k, we would perform cross-validation.



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    """YOUR CODE HERE FOR Q2"""
    print(crossValidateModalKNN(X, y, 2))
    
def splitDataSet(X, y, split=0.1):
    #check if nparray or not
    X = np.array(X)
    y = np.array(y)
    number_of_test_examples = int(len(y)*split)
    mask = np.zeros(len(y), dtype=bool)
    for i in range(number_of_test_examples):
        mask[i] = True
    X_test = X[mask]
    X_train = X[~mask]
    y_test = y[mask]
    y_train = y[~mask]
    return X_train, y_train, X_test, y_test

def crossValidateModalKNN(X, y, k):
    accuracies = []
    for i in range(10):
        model = KNN(k)
        X_train, y_train, X_test, y_test = splitDataSet(X, y)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        accuracies.append(np.sum(y_hat == y_test)/len(y_hat))
        X = np.concatenate([X_train, X_train], axis=0)
        y = np.concatenate([y_train, y_train], axis=0)
        print(i)
    return np.sum(accuracies)/10



@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    raise NotImplementedError()



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""
    raise NotImplementedError()



@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    raise NotImplementedError()



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()



if __name__ == "__main__":
    main()
