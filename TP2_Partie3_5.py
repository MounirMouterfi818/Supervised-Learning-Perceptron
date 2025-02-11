import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
data_trainMines = pd.read_csv("train_dataM.csv")
data_testMines = pd.read_csv("test_dataM.csv")
data_trainRocks = pd.read_csv("train_dataR.csv")
data_testRocks = pd.read_csv("test_dataR.csv")

# Fusionner les ensembles d'entraînement et de test
X_train_mines = data_trainMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_train_rocks = data_trainRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test_mines = data_testMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test_rocks = data_testRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))

X_combined = pd.concat([X_train_mines, X_train_rocks, X_test_mines, X_test_rocks], ignore_index=True)
y_combined = np.concatenate([np.ones(len(X_train_mines)), -1 * np.ones(len(X_train_rocks)), 
                             np.ones(len(X_test_mines)), -1 * np.ones(len(X_test_rocks))])

# Convertir en DataFrame
X_combined = pd.DataFrame(X_combined.tolist())
# Fonction de l'algorithme Pocket
# Initialisation par défaut des poids
def default_initialization(D):
    return np.zeros(D)
def perceptron_pocket(X_train, y_train, alpha, max_iter, max_errors, init_type):
    N, D = X_train.shape

    # Initialisation des poids selon le type choisi
    if init_type == "default":
        W = default_initialization(D)
    
    b = 0  # Biais
    best_W = np.copy(W)  # Meilleurs poids
    best_b = b  # Meilleur biais
    best_error_count = N  # Nombre d'erreurs minimum
    errors = []  # Liste pour stocker les erreurs par itération

    for _ in range(max_iter):
        total_error = 0
        delta_W = np.zeros(D)
        delta_b = 0

        for i in range(N):
            x_i = X_train.iloc[i]  # Exemple de donnée avec biais
            y_true = y_train[i]  # Étiquette vraie
            y_raw = np.dot(x_i, W) + b  # Calcul du produit scalaire
            y_pred = 1 if y_raw >= 0 else -1  # Prédiction

            if y_true != y_pred:
                delta_W += alpha * (y_true - y_pred) * x_i
                delta_b += alpha * (y_true - y_pred)
                total_error += 1

        W += delta_W  # Mise à jour des poids
        b += delta_b  # Mise à jour du biais
        errors.append(total_error)  # Enregistrer l'erreur

        # Sauvegarder la meilleure solution
        if total_error < best_error_count:
            best_W = np.copy(W)
            best_b = b
            best_error_count = total_error

        if total_error <= max_errors:
            break  # Arrêt si l'erreur d'apprentissage est inférieure au seuil

    return best_W, best_b, errors


# Appliquer l'algorithme du perceptron
alpha = 0.1
max_iter = 10000  # Augmenter le nombre maximum d'itérations pour tester la convergence

# Apprentissage avec l'initialisation par défaut
W_combined, b_combined, errors_combined = perceptron_pocket(X_combined, y_combined, alpha, max_iter, max_errors=0, init_type="default")

print("les poids finaux avec uniquement avec l'initialisation par défaut : ")
print(W_combined)
print("le biais final avec l'initialisation uniquement aussi par défaut :  ",b_combined)

# Vérifier le nombre d'erreurs finales
final_error_count = errors_combined[-1]

if final_error_count == 0:
    print("L'ensemble L est linéairement séparable (LS).")
else:
    print("L'ensemble L n'est pas linéairement séparable (non LS).")

# Tracer le graphique des erreurs pour visualiser la convergence
plt.plot(range(len(errors_combined)), errors_combined)
plt.xlabel("Itérations")
plt.ylabel("Nombre d'erreurs")
plt.title("Évolution des erreurs pendant l'apprentissage sur l'ensemble L")
plt.grid(True)
plt.show()
