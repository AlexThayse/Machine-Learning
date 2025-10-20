import os 
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import io, transform, util, filters, morphology
from skimage.util import random_noise
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
#%% Load dataset
df= pd.read_csv("./chinese_mnist.csv", low_memory = False)
print(df.head())

print("dataframe rows:", df.shape[0])
print("image files :", len(os.listdir("./data/")))

print('Chaque personne (suite_id) recopie 10 fois (sample_id) chaque nombre (code/value). Il y a 15 nombres differents (code), dont on donne la traduction (value).')
print('suite_id differents:', np.unique(df.suite_id))
print('nombres differents:', np.unique(df.value))

#%% Matchin image names
print('Pour chaque ligne du dataframe, il y a une image correspondante, dont le nom est : input_suite_id_sample_id_code')
def file_path_col(df):    
    file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg"
    return file_path

# Create file_path column
print('On ajoute les noms de fichier au dataframe')
df["file_path"] = df.apply(file_path_col, axis = 1)
df.head()

#%%Séparation ALEATOIRE des données en ensembles de test et d'entrainement
X=[]
def load_and_resize_image(file_path):
    img = skimage.io.imread(file_path)  
    img = skimage.transform.resize(img, (28,28,1), mode='reflect')  
    img = img.reshape(28*28)  
    X.append(img)
y = df["value"]

for file_path in df['file_path']:
    load_and_resize_image(f"./data/{file_path}")
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_random = scaler.fit_transform(X_train_random)
X_test_random = scaler.transform(X_test_random)

print("taille de X_train_random", len(X_train_random), "et taille de X_test_random", len(X_test_random))
#%% Extract subset of data - Utiliser seulement initialement
'''start = 1
end = 61
print('On ne prend que les ', end-start, ' premieres personnes \n')

# Make training set
df_train = df.loc[(df['suite_id'] >= start) &  (df['suite_id'] < end),:]
X_train_ordered = []
y_train_ordered = []
for i in range(len(df_train)):
    image = skimage.io.imread('./data/' + df_train['file_path'].values[i])
    image = skimage.transform.resize(image, (64, 64, 1), mode='reflect')
    image = image.reshape(64*64)

    X_train_ordered.append(image)
    y_train_ordered.append(df_train['value'].values[i])

print('Size of the training set:', len(df_train))

#%% Visualisation
for i in range(1, len(df_train), 100):
    plt.figure()
    image = Image.open('./data/' + df_train['file_path'].values[i])
    plt.imshow(image)
    plt.title(df_train['value'].values[i])
    plt.axis('off')

#%% Séparer en données d'entrainements et données de tes
start = 61
end = 101
print('On prend les ', end-start, ' personnes restantes \n')

# Make training set
df_test = df.loc[(df['suite_id'] >= start) &  (df['suite_id'] < end),:]

print('Size of the test set:', len(df_test))

X_test_ordered = []
y_test_ordered = []

for i in range(len(df_test)):
    image = skimage.io.imread('./data/' + df_test['file_path'].values[i])
    image = skimage.transform.resize(image, (64, 64, 1), mode='reflect')
    image = image.reshape(64 * 64)

    X_test_ordered.append(image)
    y_test_ordered.append(df_test['value'].values[i])

print('Size of the test set:', len(df_test))

#%% Standardiser les caractéristiques (mise à l'échelle des données)
scaler = StandardScaler()
X_train_ordered = scaler.fit_transform(X_train_ordered)
X_test_ordered = scaler.transform(X_test_ordered)

print("taille de X_train_ordered", len(X_train_ordered), "et taille de X_test_ordered", len(X_test_ordered))'''



#%% Classifieur KNN
train_errors = []
test_errors = []
cv_scores = []
k_values = range(1, 20)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_random, y_train_random)
   
    y_train_pred = knn.predict(X_train_random)
    y_test_pred = knn.predict(X_test_random)
   
    train_errors.append(1 - accuracy_score(y_train_random, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test_random, y_test_pred))
   
    scores = cross_val_score(knn, X_train_random, y_train_random, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"Pour k = {k}, la précision est de {accuracy_score(y_test_random,y_test_pred)}")
    print(f"La précision en validation croisée est de {scores.mean()} et la variance de {scores.var()}")
   
    print(f"k = {k}, Erreur d'entraînement : {train_errors[-1]:.4f}, Erreur de test : {test_errors[-1]:.4f}, Précision CV : {cv_scores[-1]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_errors, label="Erreur d'entraînement", marker='o')
plt.plot(k_values, test_errors, label="Erreur de test", marker='o')
plt.xlabel("Nombre de voisins (k)")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur en fonction de k pour le KNN")
plt.legend()
plt.grid(True)
plt.show()

best_k = k_values[cv_scores.index(max(cv_scores))]
print(f"Meilleur k : {best_k} avec une précision de {max(cv_scores):.4f}")

parameters = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"]
}

gridsearch = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=-1)
gridsearch.fit(X_train_random, y_train_random)
print("Meilleurs paramètres :", gridsearch.best_params_)

best_knn = gridsearch.best_estimator_
y_pred = best_knn.predict(X_test_random)
accuracy = accuracy_score(y_test_random, y_pred)
print(f"Précision du modèle optimisé : {accuracy:.4f}")
print("Rapport de classification :\n", classification_report(y_test_random, y_pred))

print("Résultats détaillés de la grille de recherche :")
print(pd.DataFrame(gridsearch.cv_results_).sort_values(by="mean_test_score", ascending=False).head())


start_time = time.time()
model_time = KNeighborsClassifier(n_neighbors=5,weights="uniform", metric="euclidean")
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")
y_scores = model_time.predict_proba(X_test_random)
y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
n_classes = y_test_binarized.shape[1]
fpr = {}
tpr = {}
roc_auc = {}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(model_time.classes_): 
    if np.any(y_test_binarized[:, i]):  
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    else:
        print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclasse (GridSearch)")
plt.legend()
plt.grid(True)
plt.show()
auc_roc_mean = np.mean(list(roc_auc.values()))
print(f"AUC-ROC moyen du modèle optimisé : {auc_roc_mean:.4f}")
cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model_time.classes_, yticklabels=model_time.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (MLP)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()


#%%Random Forest
train_errors = []
test_errors = []
cv_scores = []
n_estimators_values = [10, 50, 100, 150, 200, 250, 300, 350, 400, 700, 1000]  

for n_estimators in n_estimators_values:
    print(f"\nModèle Random Forest avec {n_estimators} arbres :\n")
   
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train_random, y_train_random)

    y_train_pred = rf_model.predict(X_train_random)
    y_test_pred = rf_model.predict(X_test_random)
  
    train_errors.append(1 - accuracy_score(y_train_random, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test_random, y_test_pred))

    scores = cross_val_score(rf_model, X_train_random, y_train_random, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"Pour nb = {n_estimators}, la précision est de {accuracy_score(y_test_random,y_test_pred)}")
    print(f"La précision en validation croisée est de {scores.mean()} et la variance de {scores.var()}")
    print(f"Précision moyenne (CV) : {scores.mean():.4f}")
    print(f"Erreur d'entraînement : {train_errors[-1]:.4f}")
    print(f"Erreur de test : {test_errors[-1]:.4f}")
    print("Rapport de classification :\n", classification_report(y_test_random, y_test_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test_random, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_values, train_errors, label="Erreur d'entraînement", marker='o')
plt.plot(n_estimators_values, test_errors, label="Erreur de test", marker='o')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("Erreur")
plt.title("Évolution des erreurs en fonction du nombre d'arbres (Random Forest)")
plt.legend()
plt.grid(True)
plt.show()

best_n_estimators = n_estimators_values[cv_scores.index(max(cv_scores))]
print(f"Meilleur n_estimators : {best_n_estimators} avec une précision moyenne de {max(cv_scores):.4f}")

param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)

print("\nGridSearchCV en cours...\n")
grid_search.fit(X_train_random, y_train_random)

print("Meilleurs paramètres :", grid_search.best_params_)
print(f"Meilleure précision (CV) : {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_random)
accuracy = accuracy_score(y_test_random, y_pred)
print(f"Précision du modèle optimisé : {accuracy:.4f}")
print("Rapport de classification :\n", classification_report(y_test_random, y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test_random, y_pred))


start_time = time.time()
model_time = RandomForestClassifier(n_estimators=350, random_state=42, max_depth=30,max_features='sqrt')
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")
y_scores = model_time.predict_proba(X_test_random)
y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
n_classes = y_test_binarized.shape[1]
fpr = {}
tpr = {}
roc_auc = {}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(model_time.classes_): 
    if np.any(y_test_binarized[:, i]):  
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    else:
        print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclasse (GridSearch)")
plt.legend()
plt.grid(True)
plt.show()
auc_roc_mean = np.mean(list(roc_auc.values()))
print(f"AUC-ROC moyen du modèle optimisé : {auc_roc_mean:.4f}")
cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model_time.classes_, yticklabels=model_time.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (MLP)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()
#%% SVM
param_grid = {
    'C': np.logspace(-3, 3, 7)  , 
    'kernel': ['linear', 'rbf', 'poly'],  
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'degree': [2, 3] 
}

grid_search = GridSearchCV(
    estimator=SVC(random_state=42), 
    param_grid=param_grid,
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1  
)

print("\nGridSearchCV en cours...\n")
grid_search.fit(X_train_random, y_train_random)

print("Meilleurs paramètres :", grid_search.best_params_)
print(f"Meilleure précision (CV) : {grid_search.best_score_:.4f}")

train_errors = []
test_errors = []
cv_scores = []
C_values = np.logspace(-3,3,10)
accuracy = []
for C in C_values:
    model = SVC(C=C, kernel='poly', degree=3, gamma=0.1)
    model.fit(X_train_random, y_train_random)

    y_train_pred = model.predict(X_train_random)
    y_test_pred = model.predict(X_test_random)
   
    scores = cross_val_score(model, X_train_random, y_train_random, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"Pour C = {C}, la précision est de {accuracy_score(y_test_random,y_test_pred)}")
    print(f"La précision en validation croisée est de {scores.mean()} et la variance de {scores.var()}")
    accuracy.append(accuracy_score(y_test_random, y_test_pred))

    train_error = 1 - accuracy_score(y_train_random, y_train_pred)
    test_error = 1 - accuracy_score(y_test_random, y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)
    print(f"Pour C = {C} la précision vaut {accuracy_score(y_test_random, y_test_pred)}")

plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_errors, label="Erreur d'entraînement", marker='o')
plt.semilogx(C_values, test_errors, label="Erreur de test", marker='o')
plt.xlabel('Valeur de C')
plt.ylabel('Erreur')
plt.title('Erreur d\'entraînement et de test en fonction de C')
plt.legend()
plt.grid(True)
plt.show()


start_time = time.time()
model_time = SVC(C=0.01, kernel='poly', degree = 3, gamma=0.1 ,probability=True)
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")
y_scores = model_time.predict_proba(X_test_random)
y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
n_classes = y_test_binarized.shape[1]
fpr = {}
tpr = {}
roc_auc = {}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(model_time.classes_): 
    if np.any(y_test_binarized[:, i]):  
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    else:
        print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclasse (GridSearch)")
plt.legend()
plt.grid(True)
plt.show()
auc_roc_mean = np.mean(list(roc_auc.values()))
print(f"AUC-ROC moyen du modèle optimisé : {auc_roc_mean:.4f}")
cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model_time.classes_, yticklabels=model_time.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (MLP)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()
#%%Bayes Naïf
var_smoothing_values = np.logspace(-9, 0, 10)

best_model = None
best_accuracy = 0
best_var_smoothing = None
cv_scores = []

for var_smoothing in var_smoothing_values:
    model = GaussianNB(var_smoothing=var_smoothing)
    print(f"Évaluation du modèle {model}...")
    model.fit(X_train_random, y_train_random)
    y_train_pred = model.predict(X_train_random)
    y_test_pred = model.predict(X_test_random)

    accuracy = accuracy_score(y_test_random, y_test_pred)
    train_error = 1 - accuracy_score(y_train_random, y_train_pred)
    test_error = 1 - accuracy_score(y_test_random, y_test_pred)
    cv_scores = []
    scores = cross_val_score(model, X_train_random, y_train_random, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())
    print(f"{model} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
    report = classification_report(y_test_random, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Rapport de Classification du modèle basique (Bayes Naïf)")
    plt.show()

    y_proba = model.predict_proba(X_test_random)
    auc_roc = roc_auc_score(y_test_random, y_proba, multi_class='ovr')
    plt.figure(figsize=(8, 6))
    fpr, tpr = {}, {}
    for i, cls in enumerate(model.classes_):  
        fpr[cls], tpr[cls], _ = roc_curve((y_test_random == cls).astype(int), y_proba[:, i])
        plt.plot(fpr[cls], tpr[cls], label=f"Classe {cls} (AUC = {roc_auc_score(y_test_random == cls, y_proba[:, i]):.4f})")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC - Modèle basique")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"AUC-ROC du modèle basique : {auc_roc:.4f}")

    cm = confusion_matrix(y_test_random, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Matrice de Confusion - Modèle basique (Bayes Naïf)")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables valeurs")
    plt.show()
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_var_smoothing = var_smoothing
        best_model = model

best_model.fit(X_train_random, y_train_random)

y_test_pred = best_model.predict(X_test_random)
y_train_pred = best_model.predict(X_train_random)

accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
print(f"Meilleure valeur de var_smoothing : {best_var_smoothing:.1e}")
print(f"Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}")

report = classification_report(y_test_random, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification du modèle optimisé (Bayes Naïf)")
plt.show()

y_proba = best_model.predict_proba(X_test_random)
auc_roc = roc_auc_score(y_test_random, y_proba, multi_class='ovr')
plt.figure(figsize=(8, 6))
fpr, tpr = {}, {}
for i, cls in enumerate(best_model.classes_):  
    fpr[cls], tpr[cls], _ = roc_curve((y_test_random == cls).astype(int), y_proba[:, i])
    plt.plot(fpr[cls], tpr[cls], label=f"Classe {cls} (AUC = {roc_auc_score(y_test_random == cls, y_proba[:, i]):.4f})")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC - Modèle optimisé")
plt.legend()
plt.grid(True)
plt.show()
print(f"AUC-ROC du modèle optimisé : {auc_roc:.4f}")

cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (Bayes Naïf)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()

start_time = time.time()
model_time = GaussianNB(var_smoothing = 1e-02)
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")
#%%Perceptron

train_errors = []
test_errors = []
cv_scores = []
eta0_values = [0.01, 0.1, 0.5, 1.0]
for eta0 in eta0_values:
    perceptron = Perceptron(eta0=eta0, random_state=42)
    print(f"Évaluation du modèle {perceptron}...")
    perceptron.fit(X_train_random, y_train_random)
    y_train_pred = perceptron.predict(X_train_random)
    y_test_pred = perceptron.predict(X_test_random)
    
    accuracy = accuracy_score(y_test_random, y_test_pred)
    train_errors.append(1 - accuracy_score(y_train_random, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test_random, y_test_pred))
    scores = cross_val_score(perceptron, X_train_random, y_train_random, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"eta0 = {eta0}, Précision : {accuracy:.4f}, Erreur d'entraînement : {train_errors[-1]:.4f}, Erreur de test : {test_errors[-1]:.4f}, Précision CV : {cv_scores[-1]:.4f}")
    report = classification_report(y_test_random, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Rapport de Classification du modèle basique pour eta0 = {eta0} (Perceptron)")
    plt.show()

    y_scores = perceptron.decision_function(X_test_random)
    y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
    n_classes = y_test_binarized.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(perceptron.classes_): 
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Multiclasse - eta0 = {eta0} (Perceptron)")
    plt.legend()
    plt.grid(True)
    plt.show()
    auc_roc_mean = np.mean(list(roc_auc.values()))
    print(f"AUC-ROC moyen du modèle pour eta0 = {eta0} : {auc_roc_mean:.4f}")
    
plt.figure(figsize=(10, 6))
plt.plot(eta0_values, train_errors, label="Erreur d'entraînement", marker='o')
plt.plot(eta0_values, test_errors, label="Erreur de test", marker='o')
plt.xlabel("eta0")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur en fonction de eta0 pour le Perceptron")
plt.legend()
plt.grid(True)
plt.show()

best_eta0 = eta0_values[cv_scores.index(max(cv_scores))]
print(f"Meilleur eta0 : {best_eta0} avec une précision de {max(cv_scores):.4f}")

parameters = {
    "max_iter": [50, 100, 200, 500, 1000],
    "eta0": [0.01, 0.1, 0.5, 1.0],
    "penalty": [None, "l2", "l1", "elasticnet"],
    "alpha": [0.0001, 0.001, 0.01],
    "shuffle": [True, False],
    "fit_intercept": [True, False]
}
gridsearch = GridSearchCV(Perceptron(random_state=42), parameters, cv=5, scoring = "accuracy", n_jobs=-1)
print(f"Évaluation du modèle {gridsearch}...")
gridsearch.fit(X_train_random, y_train_random)
print("Meilleurs paramètres :", gridsearch.best_params_)
best_perceptron = gridsearch.best_estimator_
y_test_pred = best_perceptron.predict(X_test_random)
y_train_pred = best_perceptron.predict(X_train_random)

accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(best_perceptron, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{best_perceptron} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
report = classification_report(y_test_random, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification du modèle optimisé (Perceptron)")
plt.show()

y_scores = gridsearch.best_estimator_.decision_function(X_test_random)
y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
n_classes = y_test_binarized.shape[1]
fpr = {}
tpr = {}
roc_auc = {}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(gridsearch.best_estimator_.classes_): 
    if np.any(y_test_binarized[:, i]): 
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    else:
        print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclasse (Perceptron) (Modèle optimisé)")
plt.legend()
plt.grid(True)
plt.show()
auc_roc_mean = np.mean(list(roc_auc.values()))
print(f"AUC-ROC moyen du modèle optimisé : {auc_roc_mean:.4f}")

cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_perceptron.classes_, yticklabels=best_perceptron.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (Perceptron)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()

start_time = time.time()
model_time = Perceptron(alpha = 0.0001, eta0 = 0.5, fit_intercept=True, penalty="elasticnet", max_iter=50, shuffle=True)
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")
#%%MultiLayer Perceptron

train_errors = []
test_errors = []
cv_scores = []
hidden_layer_sizes = [(128,),(256,),(512,),(256,128),(512,256),(300,150,75),(512,256,128),(1024,512,256)]       
for hidden_layer_size in hidden_layer_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=300, random_state=42)
    print(f"Évaluation du modèle {mlp}...")
    mlp.fit(X_train_random, y_train_random)
    y_train_pred = mlp.predict(X_train_random)
    y_test_pred = mlp.predict(X_test_random)
    
    accuracy = accuracy_score(y_test_random, y_test_pred)
    train_errors.append(1 - accuracy_score(y_train_random, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test_random, y_test_pred))
    scores = cross_val_score(mlp, X_train_random, y_train_random, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"hidden_layer_size = {hidden_layer_size}, Précision : {accuracy:.4f}, Erreur d'entraînement : {train_errors[-1]:.4f}, Erreur de test : {test_errors[-1]:.4f}, Précision CV : {cv_scores[-1]:.4f}")
    report = classification_report(y_test_random, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Rapport de Classification du modèle basique pour hidden_layer_size = {hidden_layer_size} (MLP)")
    plt.show()
   
    y_scores = mlp.predict_proba(X_test_random)
    y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
    n_classes = y_test_binarized.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(mlp.classes_):  
        if np.any(y_test_binarized[:, i]):  
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
            roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
            plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
        else:
            print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Multiclasse (MLP)")
    plt.legend()
    plt.grid(True)
    plt.show()
    auc_roc_mean = np.mean(list(roc_auc.values()))
    print(f"AUC-ROC moyen du modèle pour hidden_layer_size = {hidden_layer_size} : {auc_roc_mean:.4f}")

plt.figure(figsize=(10, 6))
plt.plot([str(i) for i in hidden_layer_sizes], train_errors, label="Erreur d'entraînement", marker='o')
plt.plot([str(i) for i in hidden_layer_sizes], test_errors, label="Erreur de test", marker='o')
plt.xlabel("Taille des couches cachées")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur en fonction des couches cachées pour le MLP")
plt.legend()
plt.grid(True)
plt.show()

best_hidden_layer_size = hidden_layer_sizes[cv_scores.index(max(cv_scores))]
print(f"Meilleure taille de couches cachées : {best_hidden_layer_size} avec une précision de {max(cv_scores):.4f}")

parameters = {
    "hidden_layer_sizes": [(128,),(256,),(512,),(256,128),(512,256),(300,150,75),(512,256,128),(1024,512,256)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001, 0.01,0.1],
    "learning_rate": ["constant", "adaptive"],
}
gridsearch = GridSearchCV(MLPClassifier(random_state=42), parameters, cv=5, n_jobs=-1)
print(f"Évaluation du modèle {gridsearch}...")
gridsearch.fit(X_train_random, y_train_random)
print("Meilleurs paramètres :", gridsearch.best_params_)
best_mlp = gridsearch.best_estimator_
y_test_pred = best_mlp.predict(X_test_random)
y_train_pred = best_mlp.predict(X_train_random)

accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(best_mlp, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{best_mlp} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
report = classification_report(y_test_random, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification du modèle optimisé (MLP)")
plt.show()

y_scores = gridsearch.predict_proba(X_test_random)
y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
n_classes = y_test_binarized.shape[1]
fpr = {}
tpr = {}
roc_auc = {}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(gridsearch.classes_): 
    if np.any(y_test_binarized[:, i]):  
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    else:
        print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclasse (GridSearch)")
plt.legend()
plt.grid(True)
plt.show()
auc_roc_mean = np.mean(list(roc_auc.values()))
print(f"AUC-ROC moyen du modèle optimisé : {auc_roc_mean:.4f}")

cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_mlp.classes_, yticklabels=best_mlp.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (MLP)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()

start_time = time.time()
model_time = MLPClassifier(activation = "relu",alpha = 0.1,hidden_layer_sizes=(512,256,128),learning_rate="constant",solver="adam",random_state = 42)
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = []
scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
cv_scores.append(scores.mean())
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")

#%% Logistic Regression

train_errors = []
test_errors = []
cv_scores = []
C_values = [0.001,0.01, 0.1, 1.0, 10.0, 100.0]

for C in C_values:
    logistic_reg = OneVsRestClassifier(LogisticRegression(C=C, random_state=42))
    print(f"Évaluation du modèle {logistic_reg}...")
    logistic_reg.fit(X_train_random, y_train_random)
    y_train_pred = logistic_reg.predict(X_train_random)
    y_test_pred = logistic_reg.predict(X_test_random)

    accuracy = accuracy_score(y_test_random, y_test_pred)
    train_errors.append(1 - accuracy_score(y_train_random, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test_random, y_test_pred))
    scores = cross_val_score(logistic_reg, X_train_random, y_train_random, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"C = {C}, Précision : {accuracy:.4f}, Erreur d'entraînement : {train_errors[-1]:.4f}, Erreur de test : {test_errors[-1]:.4f}, Précision CV : {cv_scores[-1]:.4f}")
    
    report = classification_report(y_test_random, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Rapport de Classification du modèle Logistic Regression pour C = {C}")
    plt.show()

    y_scores = logistic_reg.decision_function(X_test_random)
    y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
    n_classes = y_test_binarized.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(logistic_reg.classes_):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Multiclasse - C = {C} (Logistic Regression)")
    plt.legend()
    plt.grid(True)
    plt.show()
    auc_roc_mean = np.mean(list(roc_auc.values()))
    print(f"AUC-ROC moyen du modèle pour C = {C} : {auc_roc_mean:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(C_values, train_errors, label="Erreur d'entraînement", marker='o')
plt.plot(C_values, test_errors, label="Erreur de test", marker='o')
plt.xlabel("C")
plt.ylabel("Erreur")
plt.title("Évolution de l'erreur en fonction de C pour Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

best_C = C_values[cv_scores.index(max(cv_scores))]
print(f"Meilleur C : {best_C} avec une précision de {max(cv_scores):.4f}")

parameters = {
    "estimator__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  
    "estimator__penalty": ["l1", "l2", "elasticnet"],  
}

gridsearch = GridSearchCV(
    OneVsRestClassifier(LogisticRegression(random_state=42)), 
    parameters, 
    cv=5, 
    scoring="accuracy", 
    n_jobs=-1
)

print(f"Évaluation du modèle {gridsearch}...")
gridsearch.fit(X_train_random, y_train_random)
print("Meilleurs paramètres :", gridsearch.best_params_)
best_logistic_reg = gridsearch.best_estimator_
y_test_pred = best_logistic_reg.predict(X_test_random)
y_train_pred = best_logistic_reg.predict(X_train_random)

accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = cross_val_score(best_logistic_reg, X_train_random, y_train_random, cv=5, scoring="accuracy")
print(f"{best_logistic_reg} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {cv_scores.mean():.4f}")

report = classification_report(y_test_random, y_test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification du modèle optimisé (Logistic Regression)")
plt.show()

y_scores = best_logistic_reg.decision_function(X_test_random)
y_test_binarized = label_binarize(y_test_random, classes=np.unique(y_train_random))
n_classes = y_test_binarized.shape[1]
fpr = {}
tpr = {}
roc_auc = {}
plt.figure(figsize=(8, 6))
for i, cls in enumerate(best_logistic_reg.classes_):
    if np.any(y_test_binarized[:, i]): 
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_scores[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {cls} (AUC = {roc_auc[i]:.4f})')
    else:
        print(f"Classe {cls} absente dans y_test_random. ROC AUC non calculé.")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclasse (Logistic Regression) (Modèle optimisé)")
plt.legend()
plt.grid(True)
plt.show()
auc_roc_mean = np.mean(list(roc_auc.values()))
print(f"AUC-ROC moyen du modèle optimisé : {auc_roc_mean:.4f}")

cm = confusion_matrix(y_test_random, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_logistic_reg.classes_, yticklabels=best_logistic_reg.classes_)
plt.title("Matrice de Confusion - Modèle optimisé (Logistic Regression)")
plt.xlabel("Prédictions")
plt.ylabel("Véritables valeurs")
plt.show()

best_param = gridsearch.best_params_
clean_params = {key.replace("estimator__", ""): value for key, value in best_param.items()}
start_time = time.time()
model_time = LogisticRegression(**clean_params, random_state=42)
print(f"Évaluation du modèle pour déterminer le temps d'exécution {model_time}...")
model_time.fit(X_train_random, y_train_random)
y_train_pred = model_time.predict(X_train_random)
y_test_pred = model_time.predict(X_test_random)
end_time = time.time()
execution_time_nb = end_time - start_time
accuracy = accuracy_score(y_test_random, y_test_pred)
train_error = 1 - accuracy_score(y_train_random, y_train_pred)
test_error = 1 - accuracy_score(y_test_random, y_test_pred)
cv_scores = cross_val_score(model_time, X_train_random, y_train_random, cv=5, scoring="accuracy")
print(f"{model_time} - Précision : {accuracy:.4f}, Erreur d'entraînement : {train_error:.4f}, Erreur de test : {test_error:.4f}, Précision CV : {cv_scores.mean():.4f}")
print(f"Temps d'exécution : {execution_time_nb:.2f} secondes")