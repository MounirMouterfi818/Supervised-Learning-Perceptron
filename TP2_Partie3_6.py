import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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

# Fonction d'initialisation des poids par défaut
def default_initialization(D):
    return np.zeros(D)

# Fonction de l'algorithme Pocket
def perceptron_pocket(X_train, y_train, alpha, max_iter, max_errors, init_type):
    N, D = X_train.shape

    # Initialisation des poids selon le type choisi
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

# Fonction pour calculer l'erreur d'apprentissage
def calculate_learning_error(X, y, W, b):
    N = len(X)
    predictions = np.sign(np.dot(X, W) + b)
    errors = np.sum(predictions != y)
    return (errors / N) * 100  # Erreur en pourcentage

# Diviser aléatoirement les données en LA, LV et LT
def split_data_randomly(X, y, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
    indices = list(range(len(X)))
    random.shuffle(indices)
    train_end = int(len(X) * train_ratio)
    val_end = train_end + int(len(X) * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_val = X.iloc[val_indices]
    y_val = y[val_indices]
    X_test = X.iloc[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Fonction pour l'expérience d'Early Stopping
def early_stopping_experiment(X, y, alpha, max_iter, max_errors, repetitions):
    errors_train = []
    errors_val = []
    errors_test = []
    
    for _ in range(repetitions):
        # Diviser les données aléatoirement
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_randomly(X, y)
        
        # Apprentissage sur LA (X_train)
        W, b, _ = perceptron_pocket(X_train, y_train, alpha, max_iter, max_errors, init_type="default")
        
        # Erreur d'apprentissage sur LA
        Ea = calculate_learning_error(X_train, y_train, W, b)
        errors_train.append(Ea)
        
        # Erreur de validation sur LV
        Ev = calculate_learning_error(X_val, y_val, W, b)
        errors_val.append(Ev)
        
        # Erreur de test sur LT
        Et = calculate_learning_error(X_test, y_test, W, b)
        errors_test.append(Et)
    
    # Moyennes des erreurs
    mean_Ea = np.mean(errors_train)
    mean_Ev = np.mean(errors_val)
    mean_Et = np.mean(errors_test)
    
    return mean_Ea, mean_Ev, mean_Et, errors_train, errors_val, errors_test

# Répéter l'expérience plusieurs fois
alpha = 0.1
max_iter = 1000
repetitions = 10  # Nombre de répétitions

# Effectuer l'expérience d'Early Stopping
mean_Ea, mean_Ev, mean_Et, all_Ea, all_Ev, all_Et = early_stopping_experiment(
    X_combined, y_combined, alpha, max_iter, max_errors=0, repetitions=repetitions
)

# Afficher les résultats
print("\n--- Résultats Early Stopping ---")
print(f"Moyenne des erreurs d'apprentissage (Ea) : {mean_Ea:.2f}%")
print(f"Moyenne des erreurs de validation (Ev)  : {mean_Ev:.2f}%")
print(f"Moyenne des erreurs de test (Et)        : {mean_Et:.2f}%")

# Tracer l'évolution des erreurs pour chaque répétition
plt.figure(figsize=(10, 6))
plt.plot(range(1, repetitions + 1), all_Ea, label="Erreur d'apprentissage (Ea)", marker='o')
plt.plot(range(1, repetitions + 1), all_Ev, label="Erreur de validation (Ev)", marker='o')
plt.plot(range(1, repetitions + 1), all_Et, label="Erreur de test (Et)", marker='o')
plt.xlabel("Répétitions")
plt.ylabel("Erreur (%)")
plt.title("Erreurs d'apprentissage, validation et test (Early Stopping)")
plt.legend()
plt.grid(True)
plt.show()
