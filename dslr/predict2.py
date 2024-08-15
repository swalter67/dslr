import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split  # Pour diviser les données


def standard(p_data):
    df2 = pd.DataFrame()
    for col in p_data.columns:
        print(f"Processing column: {col}")
        
        # Remplacer les NaN par 0 en utilisant .loc[] pour éviter l'avertissement
        p_data.loc[:, col] = p_data[col].fillna(0)
        
        mean = p_data[col].mean()
        std = p_data[col].std()
        
        if std == 0:
            print(f"Standard deviation for column {col} is zero.")
            df2[col] = p_data[col]
        else:
            df2[col] = (p_data[col] - mean) / std
        
        assert not df2[col].isna().any(), f"NaN values detected after standardizing {col}"
    
    return df2


def predict1(g, weights):
    max_prob = (-10, 0)
    for weight, house in weights:
        if (g.dot(weight), house) > max_prob:
            max_prob = (g.dot(weight), house)
    return max_prob[1]

def predict(X, weights):
    return [predict1(i, weights) for i in np.insert(X, 0, 1, axis=1)]



def main():
    file_path = '../datasets/dataset_test.csv'
    df = pd.read_csv(file_path)

    # Séparer les données en ensemble d'entraînement et de validation
    #train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df = df
    print(valid_df)
    # Charger les poids
    weights = np.load("pred.npy", allow_pickle=True)
    
    # Sélectionner les colonnes pertinentes pour la prédiction
    pred = valid_df[["Astronomy","Herbology","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Charms","Flying"]]
    

    # Remplacer les NaN par la moyenne de chaque colonne
    for column in pred.columns:
        pred.loc[:, column] = pred[column].fillna(pred[column].mean())

    # Convertir en array numpy et standardiser
    pred = standard(pred)  # Utiliser la même fonction standard utilisée lors de l'entraînement

    # Prédictions
    predictions = predict(pred.to_numpy(), weights)

    # Obtenir les vraies maisons pour l'ensemble de validation
    true_house = valid_df["Hogwarts House"]

    

    # Créer un DataFrame avec les résultats
    results = pd.DataFrame({
        "Index": valid_df.index,
        "Hogwarts House": predictions
    })

    # Enregistrer les résultats dans un fichier CSV
    results.to_csv("house.csv", index=False)
    print("Predictions saved to house.csv")

if __name__ == "__main__":
    main()