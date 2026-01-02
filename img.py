import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Dossier contenant les images classées par dossier par classe
dataset_path = r"dataset"

# Paramètres
img_size = (64, 64)

# Charger les images et labels
X = []
y = []
classes = os.listdir(dataset_path)
class_dict = {cls_name: i for i, cls_name in enumerate(classes)}

for cls in classes:
    cls_dir = os.path.join(dataset_path, cls)
    for file in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)

            # Version pour HOG : grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray_img)
            y.append(class_dict[cls])

X = np.array(X)
y = np.array(y)

# Normalisation
X = X / 255.0

# Split pour SVM/KNN
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ============================
#        HOG FEATURES
# ============================
from skimage.feature import hog

def extract_hog_features(images):
    features = []
    for img in images:
        h = hog(img, orientations=9, pixels_per_cell=(8,8),
                cells_per_block=(2,2), block_norm='L2-Hys')
        features.append(h)
    return np.array(features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

print("Feature vector size:", X_train_hog.shape)

# ============================
#      🔹 1. SVM
# ============================
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_hog, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred_svm = svm_model.predict(X_test_hog)

print("\n===== SVM Results =====")
print("Accuracy :", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_svm, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred_svm, average='weighted'))

# ============================
#      🔹 2. KNN
# ============================
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_hog, y_train)

y_pred_knn = knn_model.predict(X_test_hog)

print("\n===== KNN Results =====")
print("Accuracy :", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_knn, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred_knn, average='weighted'))

# ============================
#  🔵 3. CNN (Modèle profond)
# ============================

# Recharger les images couleur pour le CNN
X_color = []
for cls in classes:
    cls_dir = os.path.join(dataset_path, cls)
    for file in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0
            X_color.append(img)

X_color = np.array(X_color)
y_color = np.array(y)

# Split
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_color, y_color, test_size=0.2, random_state=42, stratify=y_color
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

num_classes = len(classes)
y_train_cat = to_categorical(y_train_cnn, num_classes)
y_test_cat = to_categorical(y_test_cnn, num_classes)

cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

print("\n🔵 Entraînement du CNN...")
cnn.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)

y_pred_cnn = cnn.predict(X_test_cnn).argmax(axis=1)

print("\n===== CNN Results =====")
print("Accuracy :", accuracy_score(y_test_cnn, y_pred_cnn))
print("Precision:", precision_score(y_test_cnn, y_pred_cnn, average='weighted'))
print("Recall   :", recall_score(y_test_cnn, y_pred_cnn, average='weighted'))
print("F1 Score :", f1_score(y_test_cnn, y_pred_cnn, average='weighted'))

# ============================
#      MATRICES DE CONFUSION
# ============================

# SVM
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – SVM")
plt.show()

# KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6,5))
sns.heatmap(cm_knn, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – KNN")
plt.show()

# CNN
cm_cnn = confusion_matrix(y_test_cnn, y_pred_cnn)
plt.figure(figsize=(6,5))
sns.heatmap(cm_cnn, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – CNN")
plt.show()
