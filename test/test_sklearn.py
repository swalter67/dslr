import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Définir les étiquettes des maisons
houses = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}

# Charger les données
filepath = "../datasets/dataset_train.csv"
df = pd.read_csv(filepath)

# Sélectionner uniquement les colonnes spécifiques
_data = df[["Astronomy", "Herbology", "Divination", "Muggle Studies", 
            "Ancient Runes", "History of Magic", "Transfiguration", 
            "Potions", "Charms", "Flying"]]

# Extraire les labels
y = df['Hogwarts House']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(_data, y, random_state=20, test_size=0.20)

# Imputer les valeurs manquantes par la moyenne des colonnes
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Modèle 1 : Arbre de décision
tree = DecisionTreeClassifier() 
tree.fit(X_train, y_train) 
y_pred_tree = tree.predict(X_test) 

# Évaluation de l'arbre de décision
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree)) 
print("Precision:", precision_score(y_test, y_pred_tree, average="weighted")) 
print('F1 score:', f1_score(y_test, y_pred_tree, average="weighted")) 

# Modèle 2 : Régression logistique
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Évaluation de la régression logistique
print("\nLogistic Regression:")
accuracy_logistic = accuracy_score(y_test, y_pred_logistic) * 100
print("Accuracy:", accuracy_logistic)
print("Precision:", precision_score(y_test, y_pred_logistic, average="weighted")) 
print('F1 score:', f1_score(y_test, y_pred_logistic, average="weighted")) 

# Validation croisée sur l'arbre de décision
scores = cross_val_score(tree, _data, y, cv=5, scoring='accuracy')
print("Mean accuracy (cross-validation):", scores.mean())
