import pandas as pd
import numpy as np
import describe as ds  # Assurez-vous que ce module contient custom_mean et custom_std

def standardize(p_data):
    df2 = pd.DataFrame()
    for col in p_data.columns:
        print(f"Processing column: {col}")
        
        # Remplacer les NaN par 0 pour éviter les erreurs
        p_data.loc[:, col] = p_data[col].fillna(0)
        
        # Calcul de la moyenne et de l'écart-type personnalisés
        mean = ds.custom_mean(p_data[col])
        std = ds.custom_std(p_data[col])
        
        if std == 0:
            print(f"Standard deviation for column {col} is zero.")
            df2[col] = p_data[col]
        else:
            df2[col] = (p_data[col] - mean) / std
        
        # Vérification de l'absence de NaN après la standardisation
        assert not df2[col].isna().any(), f"NaN values detected after standardizing {col}"
    
    return df2

def sigmoid(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, -500, 500)
    return 1 / (1 + np.exp(-arr))

def sgd(X, y, learning_rate=0.001, max_iterations=1000):
    weights = np.ones(X.shape[1])
    
    for iteration in range(max_iterations):
        for i in range(X.shape[0]):
            linear_combination = np.dot(X[i], weights)
            prediction = sigmoid(linear_combination)
            error = y[i] - prediction
            gradient = X[i] * error
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
        
        if method == "sgd":
            print(f"Training {house} with Stochastic Gradient Descent")
            r = sgd(X, current_y, learning_rate=learning_rate, max_iterations=max_iterations)
        else:
            print(f"Training {house} with Gradient Descent")
            r = np.ones(X.shape[1])
            for _ in range(max_iterations):
                linear_combination = np.dot(X, r)
                predictions = sigmoid(linear_combination)
                errors = current_y - predictions
                gradient = np.dot(X.T, errors)
                r += learning_rate * gradient

        weights.append((r, house))

    return weights

def main():
    file_path = 'dataset_train.csv'
    df = pd.read_csv(file_path)
    t_data = df["Hogwarts House"]
    p_data = df[["Defense Against the Dark Arts", "Arithmancy", "Care of Magical Creatures"]]
    
    # Standardization
    p_data = standardize(p_data)
    
    # Train the model using SGD
    trained_weights = fit(p_data.to_numpy(), t_data.to_numpy(), method="sgd")

    # Save the trained weights
    np.save("pred.npy", np.array(trained_weights, dtype='object'))

    # Charger le fichier .npy
    trained_weights = np.load("pred.npy", allow_pickle=True)

    # Afficher le contenu du fichier
    print(trained_weights)

if __name__ == "__main__":
    main()
