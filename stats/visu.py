import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def plot_probabilities(X, y_probs, house_names, feature_names):
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, house in enumerate(house_names):
        plt.scatter(X[:, 0], y_probs[:, i], color=colors[i], label=house)
    
    plt.xlabel(feature_names[0])
    plt.ylabel("Predicted Probability")
    plt.title("Logistic Regression Predicted Probabilities for Each House")
    plt.legend()
    plt.show()

def main():
    file_path = 'dataset_train.csv'
    df = pd.read_csv(file_path)

    # Sélectionner les 10 colonnes de données
    features = ["Astronomy", "Herbology", "Divination", "Muggle Studies", 
                "Ancient Runes", "History of Magic", "Transfiguration", 
                "Potions", "Charms", "Flying"]
    
    X = df[features]
    y = df["Hogwarts House"]

    # Imputation des valeurs manquantes par la moyenne de chaque colonne
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Standardiser les données
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Séparer les données en ensemble d'entraînement et de validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser le modèle One-vs-All Logistic Regression pour chaque maison
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    logistic_models = {house: LogisticRegression() for house in houses}

    # Entraîner les modèles pour chaque maison
    y_probs = np.zeros((X_test.shape[0], len(houses)))
    for i, house in enumerate(houses):
        # Créer une variable cible binaire pour chaque maison (1 pour la maison, 0 pour les autres)
        y_train_binary = (y_train == house).astype(int)
        logistic_models[house].fit(X_train, y_train_binary)
        # Calculer les probabilités prédictives pour chaque maison
        y_probs[:, i] = logistic_models[house].predict_proba(X_test)[:, 1]

    # Visualiser les probabilités en fonction d'une ou deux des caractéristiques
    # Par exemple, on pourrait visualiser les probabilités par rapport à "Astronomy" et "Herbology"
    X_plot = X_test[:, :3]  # Sélectionnez les deux premières caractéristiques pour la visualisation
    plot_probabilities(X_plot, y_probs, houses, features[:3])


if __name__ == "__main__":
    main()
