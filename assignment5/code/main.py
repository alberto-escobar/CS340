#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(lr_model, X_train, y_train)
    savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    """YOUR CODE HERE FOR Q1.1"""
    # kernel logistic regression with a polynomial kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    kernel = PolynomialKernel(2)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegPolynomial.png", fig)
    
    # kernel logistic regression with a polynomial kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(0.5)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegGaussianRBF.png", fig)


@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    """YOUR CODE HERE FOR Q1.2"""
    best_train_error = float('inf')
    best_train_error_sigma = None
    best_train_error_lammy = None
    
    best_val_error = float('inf')
    best_val_error_sigma = None
    best_val_error_lammy = None
    for i in range(len(sigmas)):
        for j in range(len(lammys)):
                sigma = sigmas[i]
                lammy = lammys[j]
                loss_fn = KernelLogisticRegressionLossL2(lammy)
                optimizer = GradientDescentLineSearch()
                kernel = GaussianRBFKernel(sigma)
                klr_model = KernelClassifier(loss_fn, optimizer, kernel)
                klr_model.fit(X_train, y_train)
                train_error = np.mean(klr_model.predict(X_train) != y_train)
                val_error = np.mean(klr_model.predict(X_val) != y_val)
                if (train_error < best_train_error):
                    best_train_error = train_error
                    best_train_error_sigma = sigmas[i]
                    best_train_error_lammy = lammys[j]
                
                if (val_error < best_val_error):
                    best_val_error = val_error
                    best_val_error_sigma = sigmas[i]
                    best_val_error_lammy = lammys[j]

                train_errs[i][j] = train_error
                val_errs[i][j] = val_error

    print(f"Best Training error {best_train_error:.1%} occurs with sigma = {best_train_error_sigma:.6f} and lambda = {best_train_error_lammy:.4f}")
    
    loss_fn = KernelLogisticRegressionLossL2(best_train_error_lammy)
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(best_train_error_sigma)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)
    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegGaussianRBFBestTrainError.png", fig)

    print(f"Best Validation error {best_val_error:.1%} occurs with sigma = {best_val_error_sigma:.6f} and lambda = {best_val_error_lammy:.4f}")
    
    loss_fn = KernelLogisticRegressionLossL2(best_val_error_lammy)
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(best_val_error_sigma)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)
    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegGaussianRBFBestValError.png", fig)

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    """YOUR CODE HERE FOR Q3"""
    model = PCAEncoder(2)
    model.fit(X_train)
    W = model.W
    Z = X_train @ W.T @ np.linalg.inv(W @ W.T)
    print(Z)
    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1])
    for i in random_is:
        ax.annotate(animal_names[i], xy=Z[i, :])
    savefig("animals_answer.png", fig)
    plt.close(fig)
    trait1 = trait_names[np.argmax(np.abs(W[0,:]))]
    trait2 = trait_names[np.argmax(np.abs(W[1,:]))]

    print(trait1, trait2)

    X_centered = X_train - model.mu
    variance_explained = 1 - np.linalg.norm((Z@W - X_centered), ord='fro')**2 / np.linalg.norm((X_centered), ord='fro')**2
    print("Variance explained: {:.3f}".format(variance_explained))

    for k in range(1,100):
        model = PCAEncoder(k)
        model.fit(X_train)
        W = model.W
        Z = X_train @ W.T @ np.linalg.inv(W @ W.T)
        X_centered = X_train - model.mu
        variance_explained = 1 - np.linalg.norm((Z@W - X_centered), ord='fro')**2 / np.linalg.norm((X_centered), ord='fro')**2
        if variance_explained > 0.5:
            print("at k = {:d} the variance explained = {:.3f}".format(k, variance_explained))
            break




@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.1"""
    batch_sizes = [1,10,100]
   
    for batch_size in batch_sizes:
        loss_function = LeastSquaresLoss()
        base_optimizer = GradientDescent()
        learning_rate_getter = ConstantLR(0.0003)
        optimizer = StochasticGradient(base_optimizer, learning_rate_getter, batch_size, max_evals=10)
        model = LinearModel(loss_function, optimizer, check_correctness=False)
        model.fit(X_train, y_train)
        train_err = ((model.predict(X_train) - y_train) ** 2).mean()
        val_err = ((model.predict(X_val) - y_val) ** 2).mean()
        print("Batch size: {:d}\tTraining error: {:.3f}\tValidation error: {:.3f}".format(batch_size, train_err, val_err))


@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.3"""
    c = 0.1
    learning_rate_getters = [
            ConstantLR(c),
            InverseLR(c),
            InverseSquaredLR(c),
            InverseSqrtLR(c)
    ]
    plot_labels = [
        "constant",
        "inverse",
        "inverse_squared",
        "inverse_sqrt"
    ]
    plt.figure()
    for i in range(len(learning_rate_getters)):
        loss_function = LeastSquaresLoss()
        base_optimizer = GradientDescent()
        optimizer = StochasticGradient(base_optimizer, learning_rate_getters[i], 10, max_evals=50)
        model = LinearModel(loss_function, optimizer)
        model.fit(X_train, y_train)
        err_train = np.mean((model.predict(X_train) - y_train) ** 2)
        err_valid = np.mean((model.predict(X_val) - y_val) ** 2)
        plt.plot(model.fs, label=plot_labels[i])
    
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Objective function f value")
    savefig("learning_curves", plt)


if __name__ == "__main__":
    main()
