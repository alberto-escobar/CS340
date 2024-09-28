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
    cross_validation_accuracies = []
    test_accuracies = []
    cross_validation_training_error = []
    training_error = []
    for k in ks:
        accuracy, error = crossValidateModalKNN(X, y, k)
        cross_validation_accuracies.append(accuracy)
        cross_validation_training_error.append(error)
        model = KNN(k)
        model.fit(X, y)
        y_hat = model.predict(X_test)
        test_accuracies.append(np.mean(y_hat == y_test))

        y_hat = model.predict(X)
        training_error.append(np.mean(y_hat != y))
    plt.plot(ks, cross_validation_accuracies, label='cross validation', linestyle='-', color='blue', marker='o')
    plt.plot(ks, test_accuracies, label='test accuracy', linestyle='-', color='red', marker='x')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    plt.plot(ks, cross_validation_training_error, label='cross validation', linestyle='-', color='blue', marker='o')
    plt.plot(ks, training_error, label='training error', linestyle='-', color='red', marker='x')
    plt.xlabel('k')
    plt.ylabel('training error')
    plt.legend()
    plt.show()

    
def splitDataSet(X, y, split=0.1):
    #check if nparray or not
    X = np.array(X)
    y = np.array(y)
    number_of_test_examples = int(len(y)*split)
    mask = np.zeros(len(y), dtype=bool)
    for i in range(number_of_test_examples):
        mask[i] = True
    X_test = X[mask]
    y_test = y[mask]
    X_train = X[~mask]
    y_train = y[~mask]
    return X_train, y_train, X_test, y_test

def crossValidateModalKNN(X, y, k):
    accuracies = []
    error = []
    for i in range(10):
        model = KNN(k)
        X_train, y_train, X_test, y_test = splitDataSet(X, y)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        accuracies.append(np.mean(y_hat == y_test))
        
        y_hat = model.predict(X_train)
        error.append(np.mean(y_hat != y_train))
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
    return np.mean(accuracies), np.mean(error)


@handle("3.1")
# 3.1.1
# 1. = 0.6
# 2. = 0.4

# 3.1.2
# 1. = 1/6
# 2. = 5/6
# 3. = 2/6
# 4. = 3/4
# 5. = 1/4
# 6. = 3/4

# 3.1.3
# P(Spam|new example) = P(Spam)P(my name|spam)P(lottery|spam)(P(~venmo|spam))
# =0.6*1/6*5/6*2/6

# 3.1.4
# data set with two new rows
# [1 1 1 | 1]
# [1 1 1 | 0]

# then new calculation would be
# P(Spam|new example) = P(Spam)P(my name|spam)P(lottery|spam)(P(~venmo|spam))
# =7/12*2/7*6/7*3/7

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
    print("3.2.1 answer")
    print(wordlist[72])
    print("3.2.2 answer")
    print(np.array(wordlist)[X[802]])
    print("3.2.3 answer")
    print(groupnames[y[802]])
    #raise NotImplementedError()

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

    """CODE FOR Q3.3: Modify naive_bayes.py/NaiveBayes"""

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
    
    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

    model_lp = NaiveBayesLaplace(num_classes=4, beta=1)
    model_lp.fit(X, y)

    y_hat = model_lp.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes with laplace smoothing training error: {err_train:.3f}")

    y_hat = model_lp.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes with laplace smoothing validation error: {err_valid:.3f}")
    model_lp1k = NaiveBayesLaplace(num_classes=4, beta=1000)
    model_lp1k.fit(X, y)
    plt.plot(range(100), model.p_xy[:,0], label='naive bayes', linestyle='-', color='blue', marker='o')
    plt.plot(range(100), model_lp.p_xy[:,0], label='naive bays with laplace = 1', linestyle='-', color='red', marker='x')
    plt.plot(range(100), model_lp1k.p_xy[:,0], label='naive bays with laplace = 1000', linestyle='-', color='green', marker='.')
    plt.xlabel('j')
    plt.ylabel('P(x_ij|y_i==0)')
    plt.legend()
    plt.show()

    
@handle("3.5")
# O(t*k*d)


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
    print("Random tree")
    evaluate_model(RandomTree(max_depth=np.inf))
    # Q4.1.1 random tree has training error and does not over fit because at each stump, 
    # the feature it fits on is from a sub set of all the features in the data set. 
    # so it never chooses the optimal feature to fit at every stump since it can only fit 
    # on a sub set of the features randomly choosen 

    # Q4.1.2 it terminates eventually because the RandomTree fitting has a condition that if a stump_model has a j_best == None
    # it stops. J_best will occur if a stump is given a data set that has one example.

    print("Random forest")
    evaluate_model(RandomForest(50, np.inf))

    # Q4.1.5 a forest will fit the data perfectly, by averaging the predictions of each random tree. This results 
    # in the whole forest fitting the data perfectly and is not over fitting since it is not possible to 
    # over fit the data on the individual random trees. Using this method you achieve wisdom of the crowds.




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
    error = np.inf
    for i in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        error_i = model.error(X,y,model.means)
        if (error_i < error):
            error = error_i
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    print(f"Best error obtained: {error}")
    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")




@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    # Q5.1.1 the error goes to zero as k goes to infinity. You can never minimize the error until each cluster represents one example
    # Q5.1.2 yea probably? but I am not sure why
    # Q5.1.3 
    ks = range(1,11)
    errors = []
    for k in ks:
        error = np.inf
        for i in range(50):
            model = Kmeans(k=k)
            model.fit(X)
            y = model.predict(X)
            error_i = model.error(X,y,model.means)
            if (error_i < error):
                error = error_i
        errors.append(error)
    plt.plot(ks, errors, linestyle='-', color='blue', marker='.')
    plt.xlabel('k')
    plt.ylabel('minimum errors')
    plt.show()

                
    
    raise NotImplementedError()



if __name__ == "__main__":
    main()
