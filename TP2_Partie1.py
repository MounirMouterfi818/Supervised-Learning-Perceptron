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
y_train = np.concatenate([np.ones(len(X_train_mines)), -1 * np.ones(len(X_train_rocks))]) #les etiquette de l'ensemble d'entrainement
X_train = pd.DataFrame(X_train.tolist())  # Convertir en DataFrame
X_train['Label'] = y_train

print(X_train)

# Fonction de perceptron batch
def perceptron_batch_custom(X_train, y_train, alpha, max_iter):
    N, D = X_train.shape
    W = np.zeros(D)  # Poids
    b = 0  # Biais
    errors = []  # Stocke les erreurs par itération

    for _ in range(max_iter):
        total_error = 0
        delta_W = np.zeros(D)
        delta_b = 0

        for i in range(N):
            x_i = X_train.iloc[i]  # Caractéristiques
            y_true = y_train[i]  # Étiquette vraie
            y_raw = np.dot(x_i, W) + b
            y_pred = 1 if y_raw >= 0 else -1

            if y_true != y_pred:
                delta_W += alpha * (y_true - y_pred) * x_i
                delta_b += alpha * (y_true - y_pred)
                total_error += 1

        W += delta_W
        b += delta_b
        errors.append(total_error)

        if total_error == 0:
            break

    return W, b, errors

# Entraîner le modèle
X_train_only_features = X_train.iloc[:, :-1]
alpha = 0.1
max_iter = 1000
W, b, errors = perceptron_batch_custom(X_train_only_features, y_train, alpha, max_iter)

# Calculer l'erreur d'apprentissage
def calculate_learning_error(X_train, y_train, W, b):
    N = X_train.shape[0]
    errors = 0
    for i in range(N):
        x_i = X_train.iloc[i]
        y_true = y_train[i]
        y_raw = np.dot(x_i, W) + b
        y_pred = 1 if y_raw >= 0 else -1
        if y_pred != y_true:
            errors += 1
    Ea = (errors / N) * 100
    return Ea

Ea = calculate_learning_error(X_train_only_features, y_train, W, b)


# Préparer les données de test
X_test_mines = data_testMines['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test_rocks = data_testRocks['Values'].apply(lambda x: np.array([float(i) for i in x.split()]))
X_test = pd.concat([X_test_mines, X_test_rocks], ignore_index=True)
y_test = np.concatenate([np.ones(len(X_test_mines)), -1 * np.ones(len(X_test_rocks))]) # les etiquette de l'ensemble de test
X_test = pd.DataFrame(X_test.tolist())

# Prédire les classes pour le test
def predict(X, W, b):
    predictions = []
    for i in range(X.shape[0]):
        x_i = X.iloc[i]
        y_raw = np.dot(x_i, W) + b
        y_pred = 1 if y_raw >= 0 else -1
        predictions.append(y_pred)
    return np.array(predictions)

y_test_pred = predict(X_test, W, b)

# Calculer l'erreur de généralisation
def calculate_generalization_error(y_test_true, y_test_pred):
    errors = np.sum(y_test_true != y_test_pred)
    Eg = (errors / len(y_test_true)) * 100
    return Eg

Eg = calculate_generalization_error(y_test, y_test_pred)



# c) Calculer les stabilités des exemples de test
def calculate_stabilities(X_test, W, b):
    W_norm = np.linalg.norm(W)  # Norme des poids (sans inclure le biais)
    stabilities = []
    for i in range(X_test.shape[0]):
        x_i = X_test.iloc[i]  # Exemple de test
        distance = np.dot(x_i, W) + b  # Distance brute
        gamma = distance / W_norm  # Stabilité
        stabilities.append(gamma)
    return stabilities

stabilities = calculate_stabilities(X_test, W, b)





print("*******************************************Apprentissage sur train**************************************")
print(f"Erreur d'apprentissage (Ea): {Ea:.2f}%")
print(f"Erreur de généralisation (Eg): {Eg:.2f}%")
print("**********************************les N+1 poids de perceptron********************")
print("Poids appris (W):")
print(W)
print("Biais appris (b):", b)

# Afficher les stabilités
print("Stabilités des exemples de test (gamma):")
for i, gamma in enumerate(stabilities, 1):
    print(f"Exemple {i}: z = {gamma:.4f}")

# d) Graphique des stabilités
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(stabilities) + 1), stabilities, color='blue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Ligne séparateur (γ = 0)
plt.title("Graphique des stabilités (z) des exemples de test")
plt.xlabel("Exemples de test")
plt.ylabel("Stabilité (z)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


print("*******************************************Apprentissage sur test**************************************")
# Étape 1 : Préparer les données inversées
X_test_with_labels = X_test.copy()  # Inclure temporairement les étiquettes
X_test_with_labels['Label'] = y_test

# Réentraîner le perceptron sur l'ensemble de test
X_test_only_features = X_test_with_labels.iloc[:, :-1]
y_test_labels = X_test_with_labels['Label']
W_test, b_test, errors_test = perceptron_batch_custom(X_test_only_features, y_test_labels, alpha, max_iter)

# Calculer l'erreur d'apprentissage sur l'ensemble de test
Ea_test = calculate_learning_error(X_test_only_features, y_test_labels, W_test, b_test)


# Étape 2 : Tester sur l'ensemble d'entraînement
X_train_hidden = X_train.iloc[:, :-1]  # Enlever les étiquettes pour le test
y_train_true = y_train  # Stocker les étiquettes réelles pour calculer l'erreur de généralisation

# Prédire sur l'ensemble d'entraînement
y_train_pred = predict(X_train_hidden, W_test, b_test)

# Calculer l'erreur de généralisation
Eg_test = calculate_generalization_error(y_train_true, y_train_pred)

# Étape 3 : Calculer les stabilités des exemples de l'ensemble d'entraînement
def calculate_stabilities(X, W, b):
    """
    Calcule les stabilités gamma pour chaque exemple.
    """
    W_norm = np.linalg.norm(W)  # Norme de W
    stabilities = []
    for i in range(X.shape[0]):
        x_i = X.iloc[i]
        gamma = (np.dot(x_i, W) + b) / W_norm  # Distance signée
        stabilities.append(gamma)
    return np.array(stabilities)

stabilities_train = calculate_stabilities(X_train_hidden, W_test, b_test)

print(f"Erreur d'apprentissage sur l'ensemble de test (Ea_test) : {Ea_test:.2f}%")
print(f"Erreur de généralisation sur l'ensemble d'entraînement (Eg_test) : {Eg_test:.2f}%")
print("**********************************les N+1 poids de perceptron********************")
print("Poids appris (W):")
print(W_test)
print("Biais appris (b):", b_test)

# Afficher les valeurs des stabilités
print("\nValeurs des stabilités des exemples de l'ensemble d'entraînement :")
for idx, stability in enumerate(stabilities_train):
    print(f"Exemple {idx + 1}: Stabilité = {stability:.2f}")

# Étape 4 : Graphique des stabilités
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(stabilities_train)), stabilities_train, color='skyblue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', label='Hyperplan')
plt.title("Graphique des stabilités (ensemble d'entraînement)")
plt.xlabel("Exemple")
plt.ylabel("Stabilité (gamma)")
plt.legend()
plt.show()