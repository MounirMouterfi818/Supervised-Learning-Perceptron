import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
data_trainMines = pd.read_csv("train_dataM.csv")
data_testMines = pd.read_csv("test_dataM.csv")
data_trainRocks = pd.read_csv("train_dataR.csv")
data_testRocks = pd.read_csv("test_dataR.csv")

# Préparer les données d'entraînement
X_train_mines = data_trainMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_train_rocks = data_trainRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_train = pd.concat([X_train_mines, X_train_rocks], ignore_index=True)
y_train = np.concatenate([np.ones(len(X_train_mines)), -1 * np.ones(len(X_train_rocks))])  # Étiquettes

# Convertir en DataFrame
X_train = pd.DataFrame(X_train.tolist())  # Convertir en DataFrame
X_train['Label'] = y_train  # Ajouter la colonne d'étiquettes

# Initialisation par défaut des poids
def default_initialization(D):
    return np.zeros(D)  

# Initialisation aléatoire des poids
def random_initialization(D):
    return np.random.randn(D)  

# Initialisation de Hebb des poids
def hebb_initialization(X_train, y_train):
    N, D = X_train.shape
    W = np.zeros(D)  
    for i in range(N):
        W += y_train[i] * X_train.iloc[i].values  # Application de la règle de Hebb
    return W

# Fonction de l'algorithme Pocket
def perceptron_pocket(X_train, y_train, alpha, max_iter, max_errors, init_type):
    N, D = X_train.shape

    # Initialisation des poids selon le type choisi
    if init_type == "default":
        W = default_initialization(D)
    elif init_type == "random":
        W = random_initialization(D)
    elif init_type == "hebb":
        W = hebb_initialization(X_train, y_train)
    
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

# Calcul de l'erreur d'apprentissage
def calculate_learning_error(X, y, W, b):
    predictions = np.sign(np.dot(X, W) + b)  # Prédiction des classes
    errors = np.sum(predictions != y)  # Nombre d'erreurs
    return errors / len(y) * 100  # Retourne l'erreur en pourcentage

# Calcul de l'erreur de généralisation
def calculate_generalization_error(y_true, y_pred):
    errors = np.sum(y_true != y_pred)  # Nombre d'erreurs
    return errors / len(y_true) * 100  # Retourne l'erreur en pourcentage

# Fonction de prédiction
def predict(X, W, b):
    return np.sign(np.dot(X, W) + b)  # Retourne les prédictions (-1 ou 1)

# Apprentissage sur l'ensemble d'entraînement
X_train_only_features = X_train.iloc[:, :-1]  # Caractéristiques (sans l'étiquette)
alpha = 0.1
max_iter = 1000
max_errors = 5  # Limite d'erreurs autorisées

# Apprentissage avec l'algorithme Pocket (initialisation par défaut)
W_pocket_default, b_pocket_default, errors_pocket_default = perceptron_pocket(X_train_only_features, y_train, alpha, max_iter, max_errors, init_type="default")

# Apprentissage avec l'algorithme Pocket (initialisation aléatoire)
W_pocket_random, b_pocket_random, errors_pocket_random = perceptron_pocket(X_train_only_features, y_train, alpha, max_iter, max_errors, init_type="random")

# Apprentissage avec l'algorithme Pocket (initialisation de Hebb)
W_pocket_hebb, b_pocket_hebb, errors_pocket_hebb = perceptron_pocket(X_train_only_features, y_train, alpha, max_iter, max_errors, init_type="hebb")

# Calculer l'erreur d'apprentissage (Ea) pour l'initialisation par défaut
Ea_default = calculate_learning_error(X_train_only_features, y_train, W_pocket_default, b_pocket_default)

# Calculer l'erreur d'apprentissage (Ea) pour l'initialisation aléatoire
Ea_random = calculate_learning_error(X_train_only_features, y_train, W_pocket_random, b_pocket_random)

# Calculer l'erreur d'apprentissage (Ea) pour l'initialisation Hebb
Ea_hebb = calculate_learning_error(X_train_only_features, y_train, W_pocket_hebb, b_pocket_hebb)

# Préparer les données de test
X_test_mines = data_testMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test_rocks = data_testRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test = pd.concat([X_test_mines, X_test_rocks], ignore_index=True)
y_test = np.concatenate([np.ones(len(X_test_mines)), -1 * np.ones(len(X_test_rocks))])  # Étiquettes

# Convertir en DataFrame
X_test = pd.DataFrame(X_test.tolist())  # Convertir en DataFrame

# Prédire les classes pour le test (pour l'initialisation par défaut)
y_test_pred_pocket_default = predict(X_test, W_pocket_default, b_pocket_default)

# Prédire les classes pour le test (pour l'initialisation aléatoire)
y_test_pred_pocket_random = predict(X_test, W_pocket_random, b_pocket_random)

# Prédire les classes pour le test (pour l'initialisation de Hebb)
y_test_pred_pocket_hebb = predict(X_test, W_pocket_hebb, b_pocket_hebb)

# Calculer l'erreur de généralisation (Eg) pour l'initialisation par défaut
Eg_default = calculate_generalization_error(y_test, y_test_pred_pocket_default)

# Calculer l'erreur de généralisation (Eg) pour l'initialisation aléatoire
Eg_random = calculate_generalization_error(y_test, y_test_pred_pocket_random)

# Calculer l'erreur de généralisation (Eg) pour l'initialisation de Hebb
Eg_hebb = calculate_generalization_error(y_test, y_test_pred_pocket_hebb)

# Affichage des résultats
print("*****************Comparaison des initialisations avant echange des ensembles *****************")
print("Erreur d'apprentissage avec initialisation par défaut (Ea) :", Ea_default)
print("Erreur de generalisation avec initialisation par défaut (Eg) :", Eg_default)
print("les poids finaux avec l'initialisation par défaut : ")
print(W_pocket_default)
print("le biais final avec l'initialisation par défaut :  ",b_pocket_default)

print("\nErreur d'apprentissage avec initialisation aléatoire (Ea) :", Ea_random)
print("Erreur de generalisation avec initialisation aléatoire (Eg) :", Eg_random)

print("les poids finaux avec l'initialisation aléatoire : ")
print(W_pocket_random)
print("le biais final avec l'initialisation aléatoire :  ",b_pocket_random)

print("\nErreur d'apprentissage avec initialisation de Hebb (Ea) :", Ea_hebb)
print("Erreur de generalisation avec initialisation de Hebb (Eg) :", Eg_hebb)
print("les poids finaux avec l'initialisation de Hebb : ")
print(W_pocket_hebb)
print("le biais final avec l'initialisation de Hebb :  ",b_pocket_hebb)


# Comparaison graphique des erreurs d'apprentissage
labels = ['Initialisation Par Défaut', 'Initialisation Aléatoire', 'Initialisation Hebb']
Ea_values = [Ea_default, Ea_random, Ea_hebb]
Eg_values = [Eg_default, Eg_random, Eg_hebb]

x = np.arange(len(labels))  # position des barres
width = 0.35  # largeur des barres

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Ea_values, width, label='Erreur d\'apprentissage')
rects2 = ax.bar(x + width/2, Eg_values, width, label='Erreur de généralisation')

ax.set_ylabel('Erreurs (%)')
ax.set_title('Comparaison des erreurs avant echange des ensembles')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()

#********************************************************APRES ECHANGE DES ENSEMBLES TEST ET ENTRAINEMENT************

# Préparer les données d'entraînement (après échange des ensembles)
X_train_mines = data_testMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_train_rocks = data_testRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_train = pd.concat([X_train_mines, X_train_rocks], ignore_index=True)
y_train = np.concatenate([np.ones(len(X_train_mines)), -1 * np.ones(len(X_train_rocks))])  # Étiquettes

# Convertir en DataFrame
X_train = pd.DataFrame(X_train.tolist())  # Convertir en DataFrame
X_train['Label'] = y_train  # Ajouter la colonne d'étiquettes

# Préparer les données de test (après échange des ensembles)
X_test_mines = data_trainMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test_rocks = data_trainRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test = pd.concat([X_test_mines, X_test_rocks], ignore_index=True)
y_test = np.concatenate([np.ones(len(X_test_mines)), -1 * np.ones(len(X_test_rocks))])  # Étiquettes

# Convertir en DataFrame
X_test = pd.DataFrame(X_test.tolist())  # Convertir en DataFrame

# Apprentissage avec Pocket après échange des ensembles
X_train_only_features = X_train.iloc[:, :-1]  # Caractéristiques
alpha = 0.1
max_iter = 1000
max_errors = 5

# Initialisation par défaut
W_pocket_default, b_pocket_default, errors_pocket_default = perceptron_pocket(X_train_only_features, y_train, alpha, max_iter, max_errors, init_type="default")

# Initialisation aléatoire
W_pocket_random, b_pocket_random, errors_pocket_random = perceptron_pocket(X_train_only_features, y_train, alpha, max_iter, max_errors, init_type="random")

# Initialisation Hebb
W_pocket_hebb, b_pocket_hebb, errors_pocket_hebb = perceptron_pocket(X_train_only_features, y_train, alpha, max_iter, max_errors, init_type="hebb")

# Calculer les erreurs d'apprentissage (Ea)
Ea_default = calculate_learning_error(X_train_only_features, y_train, W_pocket_default, b_pocket_default)
Ea_random = calculate_learning_error(X_train_only_features, y_train, W_pocket_random, b_pocket_random)
Ea_hebb = calculate_learning_error(X_train_only_features, y_train, W_pocket_hebb, b_pocket_hebb)

# Prédiction sur les données de test
y_test_pred_pocket_default = predict(X_test, W_pocket_default, b_pocket_default)
y_test_pred_pocket_random = predict(X_test, W_pocket_random, b_pocket_random)
y_test_pred_pocket_hebb = predict(X_test, W_pocket_hebb, b_pocket_hebb)

# Calculer les erreurs de généralisation (Eg)
Eg_default = calculate_generalization_error(y_test, y_test_pred_pocket_default)
Eg_random = calculate_generalization_error(y_test, y_test_pred_pocket_random)
Eg_hebb = calculate_generalization_error(y_test, y_test_pred_pocket_hebb)

# Afficher les résultats
print("*****************Comparaison des initialisations apres echange des ensembles *****************")
print("Erreur d'apprentissage avec initialisation par défaut (Ea) :", Ea_default)
print("Erreur de generalisation avec initialisation par défaut (Eg) :", Eg_default)
print("les poids finaux avec l'initialisation par défaut : ")
print(W_pocket_default)
print("le biais final avec l'initialisation par défaut :  ",b_pocket_default)

print("\nErreur d'apprentissage avec initialisation aléatoire (Ea) :", Ea_random)
print("Erreur de generalisation avec initialisation aléatoire (Eg) :", Eg_random)
print("les poids finaux avec l'initialisation aléatoire : ")
print(W_pocket_random)
print("le biais final avec l'initialisation aléatoire :  ",b_pocket_random)


print("\nErreur d'apprentissage avec initialisation de Hebb (Ea) :", Ea_hebb)
print("Erreur de generalisation avec initialisation de Hebb (Eg) :", Eg_hebb)
print("les poids finaux avec l'initialisation de Hebb : ")
print(W_pocket_hebb)
print("le biais final avec l'initialisation de Hebb :  ",b_pocket_hebb)

# Comparaison graphique des erreurs d'apprentissage et de généralisation
labels = ['Par défaut', 'Aléatoire', 'Hebb']
Ea_values = [Ea_default, Ea_random, Ea_hebb]
Eg_values = [Eg_default, Eg_random, Eg_hebb]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Ea_values, width, label='Erreur d\'apprentissage')
rects2 = ax.bar(x + width/2, Eg_values, width, label='Erreur de generalisation')

ax.set_ylabel('Erreurs (%)')
ax.set_title('Comparaison des erreurs apres echange des ensembles')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()