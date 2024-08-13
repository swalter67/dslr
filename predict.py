import pandas as pd
import numpy as np
from logreg import standard


def predict1(g, weights):
    max_prob = (-10, 0)
    for weight, house in weights:
        if (g.dot(weight), house) > max_prob:
            max_prob = (g.dot(weight), house)
    return max_prob[1]


def predict(X, weights):
    return [predict1(i, weights) for i in np.insert(X, 0, 1, axis=1)]


def main():
    file_path = 'dataset_test.csv'
    df = pd.read_csv(file_path)
    weights = np.load("pred.npy", allow_pickle=True)
    pred = df[["Herbology", "Divination", "Ancient Runes", "History of Magic", "Transfiguration"]]

    # Remplacer les NaN par la moyenne de chaque colonne
    for column in pred.columns:
        pred[column] = pred[column].fillna(pred[column].mean())

    # Convertir en array numpy et standardiser
    pred = np.array(pred)
    pred = standard(pd.DataFrame(pred))  # Convertir en DataFrame pour standard

    # Prédictions
    predictions = predict(pred, weights)

    print(predictions)


if __name__ == "__main__":
    main()
