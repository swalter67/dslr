import pandas as pd
import numpy as np


def initialize_weights(dim):
    # numpy.zeros_like function return an array of zeros with the same shape
    # and type as a given array
    w = np.zeros_like(dim)
    b = 0
    return w, b


def sigmoid(z):
    return (1 / (1+np.exp(-z)))


def logloss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    log_loss = -1 * np.mean(y_true * np.log10(y_pred) + (1 - y_true) *
                            np.log10(1 - y_pred))
    return log_loss


def gradient_dw(x, y, w, b, alpha, N):
    dw = x * (y - sigmoid(np.dot(w.T, x) + b)) - ((alpha * w * w) / N)
    return dw


def gradient_db(x, y, w, b):
    db = y - sigmoid(np.dot(w.T, x) + b)
    return db


def train(X_train, y_train, X_test, y_test, iterations, alpha, eta0):
    w, b = initialize_weights(X_train[0])
    N = len(X_train)
    log_loss_train = []
    log_loss_test = []

    for i in range(0, iterations):

        for j in range(N):
            grad_dw = gradient_dw(X_train[j], y_train[j], w, b, alpha, N)
            grad_db = gradient_db(X_train[j], y_train[j], w, b)
            w = np.array(w) + (eta0 * np.array(grad_dw))
            b = b + (eta0 * grad_db)

        # predict the output of X_train[for all data points in X_train]
        # using w and b
        predict_train = []
        for m in range(len(y_train)):
            z = np.dot(w, X_train[m]) + b
            predict_train.append(sigmoid(z))

        # store all the train loss values in a list
        train_loss = logloss(y_train, predict_train)

        # predict the output of X_test[for all data points in X_test]
        # using w and b
        predict_test = []
        for m in range(len(y_test)):
            z = np.dot(w, X_test[m]) + b
            predict_test.append(sigmoid(z))

        # store all the test loss values in a list
        test_loss = logloss(y_test, predict_test)

        # we can also compare previous loss and current loss,
        # if loss is not updating then stop the process and return w,b
        if log_loss_train and train_loss > log_loss_train[-1]:
            return w, b, log_loss_train, log_loss_test

        log_loss_train.append(train_loss)
        log_loss_test.append(test_loss)

    return w, b, log_loss_train, log_loss_test
