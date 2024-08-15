import pandas as pd
import numpy as np
from logreg import standard  # Assurez-vous que cette fonction est bien définie dans logreg.py
from sklearn.model_selection import train_test_split  # Pour diviser les données

def predict1(g, weights):
    max_prob = (-10, 0)
    for weight, house in weights:
        if (g.dot(weight), house) > max_prob:
            max_prob = (g.dot(weight), house)
    return max_prob[1]

def predict(X, weights):
    return [predict1(i, weights) for i in np.insert(X, 0, 1, axis=1)]

def calcul_accuracy(predictions, true_labels):
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return correct / len(true_labels)

def main():
    file_path = 'dataset_train.csv'
    df = pd.read_csv(file_path)

    # Séparer les données en ensemble d'entraînement et de validation
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Charger les poids
    weights = np.load("pred.npy", allow_pickle=True)
    
    # Sélectionner les colonnes pertinentes pour la prédiction
    pred = valid_df[["Astronomy", "Herbology", "Divination", "Muggle Studies", 
                     "Ancient Runes", "History of Magic", "Transfiguration", 
                     "Potions", "Charms", "Flying", 
                     "Defense Against the Dark Arts", "Arithmancy", 
                     "Care of Magical Creatures"]]

    # Remplacer les NaN par la moyenne de chaque colonne
    for column in pred.columns:
        pred.loc[:, column] = pred[column].fillna(pred[column].mean())

    # Convertir en array numpy et standardiser
    pred = standard(pred)  # Utiliser la même fonction standard utilisée lors de l'entraînement

    # Prédictions
    predictions = predict(pred.to_numpy(), weights)

    # Obtenir les vraies maisons pour l'ensemble de validation
    true_house = valid_df["Hogwarts House"]

    # Calculer l'accuracy
    accuracy = calcul_accuracy(predictions, true_house)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
