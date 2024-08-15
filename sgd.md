La descente de gradient stochastique (SGD, pour **Stochastic Gradient Descent**) est une variante de l'algorithme de descente de gradient, qui est un algorithme d'optimisation utilisé pour minimiser une fonction coût en ajustant les paramètres du modèle.

### Rappel sur la Descente de Gradient
La **descente de gradient** est une méthode pour trouver le minimum d'une fonction en suivant le gradient de la fonction dans la direction opposée. Dans le contexte du machine learning, cette fonction est généralement une fonction de coût ou de perte qui mesure à quel point le modèle est incorrect. Le but de la descente de gradient est de trouver les paramètres (ou poids) qui minimisent cette fonction de coût.

- **Formule de mise à jour des poids :**
  \[
  \theta = \theta - \eta \cdot \nabla J(\theta)
  \]
  où :
  - \(\theta\) sont les paramètres (ou poids) du modèle.
  - \(\eta\) est le taux d'apprentissage (learning rate).
  - \(\nabla J(\theta)\) est le gradient de la fonction de coût \(J(\theta)\) par rapport aux paramètres.

### Descente de Gradient Stochastique
La **descente de gradient stochastique** diffère de la descente de gradient classique principalement par la manière dont elle traite les données :

- **Descente de Gradient Classique (Batch Gradient Descent)** :
  - Utilise l'ensemble complet des données pour calculer le gradient.
  - Cela signifie que pour chaque mise à jour des paramètres, tous les exemples de données sont utilisés pour calculer le gradient moyen.
  - Peut être très lent et coûteux en termes de calcul lorsque le nombre d'exemples est très grand.

- **Descente de Gradient Stochastique (SGD)** :
  - Au lieu de calculer le gradient basé sur l'ensemble complet des données, le SGD calcule le gradient en utilisant **un seul exemple** à la fois (ou un petit sous-ensemble aléatoire appelé mini-batch).
  - Après avoir calculé le gradient pour cet exemple, les paramètres sont mis à jour immédiatement.
  - L'itération sur l'ensemble des données une fois est appelée une époque.

### Avantages du SGD

1. **Rapidité :**
   - Parce qu'il met à jour les paramètres après chaque exemple, le SGD est souvent beaucoup plus rapide que la descente de gradient classique pour les grands ensembles de données.

2. **Moins de Mémoire :**
   - Il nécessite moins de mémoire, car il ne traite qu'un seul exemple à la fois.

3. **Évasion des Minima Locaux :**
   - Le fait que les gradients soient calculés à partir d'exemples individuels (qui sont bruités) signifie que le SGD peut parfois échapper aux minima locaux d'une manière que la descente de gradient classique ne peut pas.

### Inconvénients du SGD

1. **Variabilité des Mises à Jour :**
   - Les mises à jour peuvent être très variables d'une itération à l'autre, ce qui peut rendre l'entraînement du modèle moins stable.

2. **Difficulté de Convergence :**
   - Le SGD peut osciller autour du minimum global plutôt que de converger directement vers celui-ci. Cela peut être atténué en réduisant progressivement le taux d'apprentissage (learning rate) au fil des époques.

