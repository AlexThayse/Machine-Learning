import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_mnist_from_directory(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            image = Image.open(img_path).convert('L') 
            X.append(np.array(image).flatten())
           
            label = int(filename.split('_')[-1].split('.')[0])
            y.append(label)
    return np.array(X), np.array(y)

path_to_directory = 'C:/Users/manon/OneDrive/GoodNotes/Bac 3/Machine Learning/Projet/data/donnees_augmentees_sans_classe1000/'
X, y = load_mnist_from_directory(path_to_directory)

#%% Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print("X_train", len(X_train))

#%% Standardize the data
scaler = StandardScaler()
X_train_augmented = scaler.fit_transform(X_train)
X_test_augmented = scaler.transform(X_test)

#%% ACP
pca = PCA()
pca.fit(X_train_augmented)

explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)

n_components = np.argmax(explained_variance_cumsum >= 0.95) + 1
print(f"Nombre de composants pour 95% de variance expliquée : {n_components}")

pca = PCA(n_components=n_components)
X_train = pca.fit_transform(X_train_augmented)
X_test = pca.transform(X_test_augmented)

#%% Méthodes d'ensemble - Voting Classifier
from sklearn.ensemble import VotingClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='uniform')
RF = RandomForestClassifier(n_estimators=350, random_state=42, max_depth=None, max_features='sqrt')
SVM = SVC(C=0.01, kernel='poly', gamma=0.1, degree = 3, probability=True)
MLP = MLPClassifier(activation='relu', alpha=0.1, hidden_layer_sizes=(300, 150, 75))
RL = LogisticRegression(C = 0.01, penalty = 'l2')
Perceptron = Perceptron(max_iter=50, eta0=0.5, fit_intercept = True, alpha = 0.0001, penalty ='elasticnet', shuffle = True, tol = 1e-05)

ensemble_model = VotingClassifier(estimators=[
    ('RF', RF),
    ('MLP', MLP),
    ('knn',knn),
    ('SVM', SVM)
   
], voting='soft')  

ensemble_model.fit(X_train, y_train)
ensemble_y_pred = ensemble_model.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_y_pred)
print(f"Ensemble Model (Voting): Accuracy = {ensemble_acc:.4f}")


#%% Bagging/Boosting
'''
models = {
    'KNN': knn,
    'Bayes Naif': bayes_naif,
    'Perceptron': perceptron,
    'RL': RL,
}

bagging_results = {}
boosting_results = {}

# Bagging
print("\n\033[1mBagging Results:\033[0m")
for name, model in models.items():
    bagging = BaggingClassifier(estimator=model, n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bagging_results[name] = acc
    print(f"{name}: Accuracy = {acc:.4f}")

# Boosting
print("\n\033[1mBoosting Results:\033[0m")
for name, model in models.items():
    try:
        boosting = AdaBoostClassifier(estimator=model, n_estimators=10, random_state=42)
        boosting.fit(X_train, y_train)
        y_pred = boosting.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        boosting_results[name] = acc
        print(f"{name}: Accuracy = {acc:.4f}")
    except Exception as e:
        print(f"{name}: Boosting not applicable ({str(e)})")

# Résultats finaux
print("\n\033[1mSummary of Results:\033[0m")
print("Bagging:")
for name, acc in bagging_results.items():
    print(f"{name}: {acc:.4f}")

print("\nBoosting:")
for name, acc in boosting_results.items():
    print(f"{name}: {acc:.4f}")'''
   
#%% Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

base_learners = [
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='uniform')),
    ('rf', RandomForestClassifier(n_estimators=350, random_state=42, max_depth=None, max_features='sqrt')),
    ('svm', SVC(C=0.01, kernel='poly', gamma=0.1, degree=3, probability=True)),
    ('mlp', MLPClassifier(activation='relu', alpha=0.1, hidden_layer_sizes=(300, 150, 75)))
]

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=350, random_state=42, max_depth=None, max_features='sqrt')),
    ('mlp', MLPClassifier(activation='relu', alpha=0.1, hidden_layer_sizes=(512, 256, 128))),
    ('svm', SVC(C=0.01, kernel='poly', gamma=0.1, degree=3, probability=True)),
]


meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_augmented = scaler.fit_transform(X_train)
X_test_augmented = scaler.transform(X_test)

pca = PCA(n_components=260)  
X_train_pca = pca.fit_transform(X_train_augmented)
X_test_pca = pca.transform(X_test_augmented)

stacking_model.fit(X_train_pca, y_train)

stacking_y_pred = stacking_model.predict(X_test_pca)

stacking_acc = accuracy_score(y_test, stacking_y_pred)
print(f"Stacking Model Accuracy: {stacking_acc:.4f}")

