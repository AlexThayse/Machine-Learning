import os 
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import io, transform, util, morphology
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv("./chinese_mnist.csv", low_memory=False)

df = df[df['value'] != 1000]
print(f"Données après exclusion de la classe 9 : {df.shape[0]} lignes restantes.")

def file_path_col(row):
    return f"input_{row[0]}_{row[1]}_{row[2]}.jpg"

df["file_path"] = df.apply(file_path_col, axis=1)

code_to_value = dict(zip(df.drop_duplicates(subset=['code'])['code'], df['value']))

base_dir = "./data/"
augmented_data_dir = os.path.join(base_dir, "donnees_augmentees_sans_classe1000")
os.makedirs(augmented_data_dir, exist_ok=True)

def modify_thickness(image, mode='thicken'):
    if mode == 'thicken':
        return morphology.dilation(image, morphology.disk(1))
    elif mode == 'thin':
        return morphology.erosion(image, morphology.disk(1))
    return image

def augment_image(image):
    augmented_images = []

    for angle in [-10, 10]:
        rotated = transform.rotate(image, angle, mode='wrap')
        augmented_images.append(rotated)

    for shift in [(5, 5), (-5, -5)]:
        translated = transform.warp(image, transform.AffineTransform(translation=shift), mode='wrap')
        augmented_images.append(translated)

    for scale in [0.8, 1.2]:
        zoomed = transform.rescale(image, scale, mode='reflect')
        if scale < 1.0:
            pad_width = (image.shape[0] - zoomed.shape[0]) // 2
            zoomed = np.pad(zoomed, pad_width=((pad_width, pad_width), (pad_width, pad_width)), mode='constant')
        else:
            crop_width = (zoomed.shape[0] - image.shape[0]) // 2
            zoomed = zoomed[crop_width:-crop_width, crop_width:-crop_width]
        augmented_images.append(transform.resize(zoomed, image.shape))

    for shear in [0.1, -0.1]:
        distorted = transform.warp(image, transform.AffineTransform(shear=shear), mode='wrap')
        augmented_images.append(distorted)

    for mode in ['thicken', 'thin']:
        thickened_or_thinned = modify_thickness(image, mode)
        augmented_images.append(thickened_or_thinned)

    return augmented_images

print("Classes présentes dans le DataFrame après exclusion :", df['value'].unique())

def save_augmented_images(file_paths, labels):
    augmented_data = []
    augmented_labels = []

    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        img = io.imread(file_path, as_gray=True)
        img = transform.resize(img, (28, 28), mode='reflect')
       
        assert label != 1000, f"Label 1000 détecté au chemin : {file_path}"

        orig_file_name = f"img_orig_{idx}_{label}.jpg"
        orig_file_path = os.path.join(augmented_data_dir, orig_file_name)
        if not os.path.exists(orig_file_path):
            io.imsave(orig_file_path, util.img_as_ubyte(img))
            augmented_data.append(img)
            augmented_labels.append(label)

        augmented_images = augment_image(img)
        for aug_idx, aug_img in enumerate(augmented_images):
            aug_file_name = f"aug_{idx}_{aug_idx}_{label}.jpg"
            aug_file_path = os.path.join(augmented_data_dir, aug_file_name)
            if not os.path.exists(aug_file_path):
                io.imsave(aug_file_path, util.img_as_ubyte(aug_img))
                augmented_data.append(aug_img)
                augmented_labels.append(label)

    return augmented_data, augmented_labels

X_train_augmented, y_train_augmented = save_augmented_images(
    [f"./data/{fp}" for fp in df['file_path']],
    df['value']
)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

print("Classes présentes après augmentation :", np.unique(y_train_augmented))
print(f"Nombre total d'images augmentées : {len(X_train_augmented)}")
print(f"Length of augmented labels: {len(y_train_augmented)}")

#%%Séparation ALEATOIRE des données en ensembles de test et d'entrainement
def load_and_resize_image(file_path):
    try:
        img = skimage.io.imread(file_path, as_gray=True)  
        img = skimage.transform.resize(img, (28, 28), mode='reflect') 
        return img.flatten()  
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {file_path}: {e}")
        return None

def load_images_in_batches(file_paths, labels, batch_size=1000):
    """
    Charge les images par lots à partir des chemins de fichiers et associe les labels.
   
    Args:
        file_paths (list): Liste des chemins de fichiers d'images.
        labels (list): Liste des labels associés aux fichiers.
        batch_size (int): Taille du lot.
       
    Yields:
        tuple: Deux tableaux numpy, X_batch (images) et y_batch (labels).
    """
    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]  
       
        X_batch = []
        y_batch = []
       
        for file_path, label in zip(batch_files, batch_labels):
            img = load_and_resize_image(file_path)
            if img is not None:
                X_batch.append(img)  
                y_batch.append(label) 
       
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
       
        yield X_batch, y_batch

augmented_file_paths = [os.path.join(augmented_data_dir, f) for f in os.listdir(augmented_data_dir)]

y_train_augmented = [int(f.split('_')[3].split('.')[0]) for f in os.listdir(augmented_data_dir) if f.endswith('.jpg')]


print("Classes présentes dans y_test_augmented :", np.unique(y_train_augmented))

assert len(augmented_file_paths) == len(y_train_augmented), \
    f"Nombre de fichiers ({len(augmented_file_paths)}) et de labels ({len(y_train_augmented)}) ne correspond pas !"

X = []
y = []

for X_batch, y_batch in load_images_in_batches(augmented_file_paths, y_train_augmented, batch_size=1000):
    X.extend(X_batch)
    y.extend(y_batch)

X = np.array(X)
y = np.array(y)

print(f"Nombre total d'images (X) : {len(X)}")
print(f"Nombre total de labels (y) : {len(y)}")
assert len(X) == len(y), "Le nombre d'images et de labels doit être le même !"

print(f"Nombre de fichiers : {len(augmented_file_paths)}")
print(f"Nombre de labels : {len(y_train_augmented)}")
assert len(augmented_file_paths) == len(y_train_augmented), \
    f"Erreur : {len(augmented_file_paths)} fichiers et {len(y_train_augmented)} labels ne correspondent pas !"


#%% Séparation des données en ensembles d'entraînement et de test
if len(X_train_augmented) == 0 or len(y_train_augmented) == 0:
    print("Erreur : Aucune image augmentée n'a été générée.")
else:
    X_train_augmented, X_test_augmented, y_train_augmented, y_test_augmented = train_test_split(
        X, y, test_size=0.08, random_state=42, stratify=y
    )

scaler = StandardScaler()
X_train_augmented = scaler.fit_transform(X_train_augmented)
X_test_augmented = scaler.transform(X_test_augmented)

print("Taille de X_train :", len(X_train_augmented))
print("Taille de X_test :", len(X_test_augmented))

#%% Affichage des premières images
def display_images(images, n_images=5):
    """Affiche les premières n_images images d'un jeu de données."""
    plt.figure(figsize=(15, 15))

    for i in range(min(n_images, len(images))):
        plt.subplot(1, n_images, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')  
        plt.axis('off') 
        plt.title(f'Image {i+1}')
   
    plt.show()

display_images(X_train_augmented[:5])

#%% Appliquer PCA
pca = PCA()
pca.fit(X_train_augmented)

explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance_cumsum) + 1), explained_variance_cumsum, marker='o')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée')
plt.title('Variance expliquée cumulée en fonction du nombre de composantes principales')
plt.grid(True)
plt.show()

n_components = np.argmax(explained_variance_cumsum >= 0.95) + 1
print(f"Nombre de composants pour 95% de variance expliquée : {n_components}")

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_augmented)
X_test_pca = pca.transform(X_test_augmented)

#%%Test sur KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="manhattan")
knn.fit(X_train_pca, y_train_augmented)

y_pred = knn.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (K-nn)")
plt.show()

precisionKNN = accuracy_score(y_test_augmented, y_pred)
print("précision KNN = ", precisionKNN)

#%% Test sur Random Forest
n = 350
rf_model = RandomForestClassifier(n_estimators=n, random_state=42, max_depth=None, max_features='sqrt')
rf_model.fit(X_train_pca, y_train_augmented)

y_pred = rf_model.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (Random Forest)")
plt.show()

precisionRF = accuracy_score(y_test_augmented, y_pred)
print("précision RF = ", precisionRF)

#%% Test sur SVM
C = 0.01
model = SVC(C=C, kernel='poly', gamma=0.1, degree = 3, probability=True)
model.fit(X_train_pca, y_train_augmented)

y_pred = model.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (SVM)")
plt.show()

precisionSVM = accuracy_score(y_test_augmented, y_pred)
print("précision SVM = ", precisionSVM)

#%% Test sur Bayes Naif
var_smoothing = 0.01
model = GaussianNB(var_smoothing=var_smoothing)
model.fit(X_train_pca, y_train_augmented)

y_pred = model.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (NB)")
plt.show()

precisionNB = accuracy_score(y_test_augmented, y_pred)
print("précision NB = ", precisionNB)

#%% Test sur Perceptron
max_iter = 50
model = Perceptron(max_iter=max_iter, eta0=0.5, fit_intercept = True, alpha = 0.0001, penalty ='elasticnet', shuffle = True, tol = 1e-05)
model.fit(X_train_pca, y_train_augmented)

y_pred = model.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (Perceptron)")
plt.show()

precisionP = accuracy_score(y_test_augmented, y_pred)
print("précision Perceptron = ", precisionP)

#%% Test sur MultiLayer Perceptron
modelMLP = MLPClassifier(activation='relu', alpha=0.1, hidden_layer_sizes=(300, 150, 75))
modelMLP.fit(X_train_pca, y_train_augmented)
y_pred = modelMLP.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (MultiLayer Perceptron)")
plt.show()

precisionMLP = accuracy_score(y_test_augmented, y_pred)
print("précision Multilayer Perceptron = ", precisionMLP)

#%% Test sur régression logistique
model = LogisticRegression(C = 0.01, penalty = 'l2')
model.fit(X_train_pca, y_train_augmented)
y_pred = model.predict(X_test_pca)
report = classification_report(y_test_augmented, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification (Regression Logistique)")
plt.show()

precisionRL = accuracy_score(y_test_augmented, y_pred)
print("précision Regression Logistique = ", precisionRL)
