import pandas as pd
import numpy as np
import argparse

from utils import standardize, calcul_accuracy, predict, sigmoid


def hypothesis(X, r):
    return np.dot(X, r)


def grad(X, y, r):
    linear_combination = hypothesis(X, r)
    predictions = sigmoid(linear_combination)
    errors = y - predictions
    gradient = np.dot(X.T, errors)
    return gradient, predictions


def create_mini_batches(X, y, batch_size=32):
    mini_batches = []
    y = y.reshape(-1, 1)
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size: (i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size: data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


def mini_batch_gd(X, y, learning_rate=0.001, max_iterations=1000, batch_size=32):
    weights = np.ones(X.shape[1])
    weights = weights.reshape(-1, 1)

    for _ in range(max_iterations):

        mini_batches = create_mini_batches(X, y, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch

            gradient, _ = grad(X_mini, y_mini, weights)

            assert not np.isnan(gradient).any(), "Gradient contains NaN values"
            assert not np.isinf(gradient).any(), "Gradient contains infinite values"

            weights += learning_rate * gradient

    return weights


def sgd(X, y, learning_rate=0.001, max_iterations=1000):
    weights = np.ones(X.shape[1])

    for iteration in range(max_iterations):
        for i in range(X.shape[0]):
            gradient, prediction = grad(X[i], y[i], weights)
            weights += learning_rate * gradient

        if iteration % 1000 == 0:
            loss = np.mean(-y * np.log(prediction) - (1 - y) * np.log(1 - prediction))
            print(f"Iteration {iteration}: Loss = {loss}")

    return weights


def fit(X, y, method="gd"):
    weights = []
    max_iterations = 1000
    learning_rate = 0.001
    X = np.insert(X, 0, 1, axis=1)  # Add bias term

    # Vérification de NaN et Inf avant l'entraînement
    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isinf(X).any(), "X contains infinite values"

    for house in np.unique(y):
        current_y = np.where(y == house, 1, 0)

        if method == "-sgd":
            print(f"Training {house} with Stochastic Gradient Descent")
            r = sgd(X, current_y, learning_rate=learning_rate, max_iterations=max_iterations)
        elif method == "-gd":
            print(f"Training {house} with Gradient Descent")
            r = np.ones(X.shape[1])
            for _ in range(max_iterations):
                gradient, _ = grad(X, current_y, r)
                r += learning_rate * gradient
        elif method == "-mb":
            print(f"Training {house} with Mini-batch Gradient Descent")
            r = mini_batch_gd(X, current_y, learning_rate=learning_rate, max_iterations=max_iterations)

        else:
            raise ValueError(f"Unknown method: {method}")

        weights.append((r, house))
    return weights


def main():
    parser = argparse.ArgumentParser(argument_default='-gd') 
    parser.add_argument("-gd", dest="method", action="store_const", const="-gd",
                        help="The batch gradient descent algorithm \
                            will be used to train the model. \
                                It is the algorithm used by default.")
    parser.add_argument("-sgd", dest="method", action="store_const", const="-sgd",
                        help="The stochastic gradient descent \
                        algorithm will be used to train the model.")
    parser.add_argument("-mb", dest="method", action="store_const", const="-mb",
                        help="The mini-batch gradient descent \
                        algorithm will be used to train the model.")
    args = parser.parse_args()

    try:
        file_path = '../datasets/dataset_train.csv'
        df = pd.read_csv(file_path)
        t_data = df["Hogwarts House"]
        p_data = df[["Astronomy","Herbology","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]]

        # Standardization
        p_data = standardize(p_data)

        # Train the model using SGD
        trained_weights = fit(p_data.to_numpy(), t_data.to_numpy(), args.method)

        # Save the trained weights
        np.save("pred.npy", np.array(trained_weights, dtype='object'))

        # Charger le fichier .npy
        trained_weights = np.load("pred.npy", allow_pickle=True)

        # Prédire avec les poids entraînés
        predictions = predict(p_data.to_numpy(), trained_weights)

        # Calculer l'accuracy
        accuracy = calcul_accuracy(predictions, t_data.to_numpy())
        print("Accuracy:", accuracy)

        # Afficher le contenu du fichier
        # print(trained_weights)
    except Exception as e:
        print(f'An error has occurred: { e }')


if __name__ == "__main__":
    main()
