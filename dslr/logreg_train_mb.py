try:
    import sys
    import os
    import pandas as pd
    import numpy as np

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import stats.describe as ds
except ImportError:
    print("Some libraries are missing. You can install them by typing:")
    print("pip install <library>")
    exit(1)


def standardize(p_data):
    df2 = pd.DataFrame()
    for col in p_data.columns:
        print(f"Processing column: {col}")

        # Remplacer les NaN par 0 en utilisant .loc[] pour éviter l'avertissement
        p_data.loc[:, col] = p_data[col].fillna(0)

        mean = ds.custom_mean(p_data[col])
        std = ds.custom_std(p_data[col])

        if std == 0:
            print(f"Standard deviation for column {col} is zero.")
            df2[col] = p_data[col]
        else:
            df2[col] = (p_data[col] - mean) / std

        assert not df2[col].isna().any(), f"NaN values detected after standardizing {col}"

    return df2


def sigmoid(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, -500, 500)
    return 1 / (1 + np.exp(-arr))


# def gradient_descent(X, h, y):
#     return np.dot(X.T, (h - y)) / y.shape[0]


def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient


def hypothesis(X, r):
    return np.dot(X, r)


def grad(X, y, r):
    linear_combination = hypothesis(X, r)
    predictions = sigmoid(linear_combination)
    errors = y - predictions
    gradient = np.dot(X.T, errors)
    return gradient


def create_mini_batches(X, y, batch_size):
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


def fit(X, y, batch_size=32):
    weights = []
    max_iterations = 100
    learning_rate = 0.001
    X = np.insert(X, 0, 1, axis=1)

    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isinf(X).any(), "X contains infinite values"

    for house in np.unique(y):
        current_y = np.where(y == house, 1, 0)
        r = np.ones(X.shape[1])
        r = r.reshape(-1, 1)

        for _ in range(max_iterations):

            mini_batches = create_mini_batches(X, current_y, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch

                gradient = grad(X_mini, y_mini, r)

                assert not np.isnan(gradient).any(), "Gradient contains NaN values"
                assert not np.isinf(gradient).any(), "Gradient contains infinite values"

                r += learning_rate * gradient

        weights.append((r, house))

    return weights


def main():

    try:
        file_path = '../datasets/dataset_train.csv'
        df = pd.read_csv(file_path)
        t_data = df["Hogwarts House"]
        # p_data = df[["Herbology", "Divination", "Ancient Runes", "Charms", "Defense Against the Dark Arts"]]
        p_data = df[["Astronomy","Herbology","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]]
        # Standardization
        p_data = standardize(p_data)

        # Train the model
        trained_weights = fit(p_data.to_numpy(), t_data.to_numpy())

        # Save the trained weights
        np.save("pred.npy", np.array(trained_weights, dtype='object'))

        # Charger le fichier .npy
        trained_weights = np.load("pred.npy", allow_pickle=True)

        #    Afficher le contenu du fichier
        print(trained_weights)

    except Exception as e:
        print(f'An error has occurred: { e }')


if __name__ == "__main__":
    main()