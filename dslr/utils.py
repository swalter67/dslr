try:
    import sys
    import os
    import pandas as pd
    import numpy as np

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import stats.describe as ds
except ImportError:
    print("Necessary libraries are not installed. \
          Please run `pip install -r requirements.txt`")


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


def calcul_accuracy(predictions, true_labels):
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return correct / len(true_labels)


def predict(X, weights):
    # Ajouter une colonne de 1 pour le biais
    X = np.insert(X, 0, 1, axis=1)

    predictions = []
    for row in X:
        probs = []
        for w, house in weights:
            prob = sigmoid(np.dot(row, w))
            probs.append((prob, house))
        predicted_house = max(probs)[1]
        predictions.append(predicted_house)
    return predictions
