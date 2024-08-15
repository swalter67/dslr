import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Fonction standard pour la standardisation (si vous l'avez dans logreg.py)
def standard(p_data):
    df2 = pd.DataFrame()
    for col in p_data.columns:
        mean = p_data[col].mean()
        std = p_data[col].std()
        if std == 0:
            df2[col] = p_data[col]
        else:
            df2[col] = (p_data[col] - mean) / std
    return df2

# Fonction pour faire une prédiction sur un exemple
def predict1(g, weights):
    max_prob = (-10, 0)  # Valeur de départ pour max_prob
    for weight, house in weights:
        if (g.dot(weight), house) > max_prob:
            max_prob = (g.dot(weight), house)
    return max_prob[1]

# Fonction pour faire des prédictions sur un ensemble de données
def predict(X, weights):
    return [predict1(i, weights) for i in np.insert(X, 0, 1, axis=1)]

# Fonction pour calculer l'accuracy
def calcul_accuracy(predictions, true_labels):
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return correct / len(true_labels)

def main():
    file_path = 'dataset_train.csv'
    df = pd.read_csv(file_path)

    # Séparer les données en ensemble d'entraînement et de validation
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    # Charger les poids du modèle
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

    # Standardiser les données
    pred = standard(pred)

    # Convertir en numpy array
    pred = pred.to_numpy()

    # Faire des prédictions
    predictions = predict(pred, weights)

    # Obtenir les vraies maisons pour l'ensemble de validation
    true_house = valid_df["Hogwarts House"]

    # Calculer l'accuracy
    accuracy = calcul_accuracy(predictions, true_house)
    print("Accuracy:", accuracy)

    # Calculer et afficher la matrice de confusion sous forme de heatmap
    conf_matrix = confusion_matrix(true_house, predictions, labels=["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"])
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"], 
                yticklabels=["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

if __name__ == "__main__":
    main()
