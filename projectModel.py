import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
from tkinter import Tk, filedialog
from PIL import Image
import gradio as gr

train_dir = r"C:\Users\ebrah\Downloads\archive\chest_xray\train"
val_dir   = r"C:\Users\ebrah\Downloads\archive\chest_xray\val"
test_dir  = r"C:\Users\ebrah\Downloads\archive\chest_xray\test"

IMG_SIZE = 224
BATCH = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

base_model.trainable = True
for layer in base_model.layers[:-30]:  # freeze all except last 30 layers
    layer.trainable = False


categories = os.listdir(train_dir)

def extract_features(generator):
    features = []
    labels = []
    for batch_imgs, batch_labels in generator:
        feats = base_model.predict(batch_imgs, verbose=0)
        features.append(feats)
        labels.append(batch_labels.argmax(axis=1))
        if len(features) * BATCH >= generator.samples:
            break
    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features(train_gen)
X_val, y_val     = extract_features(val_gen)
X_test, y_test     = extract_features(test_gen)

svm = SVC(kernel='poly', degree=4, C=10.0, gamma='scale')

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("SVM Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=train_gen.class_indices.keys()))

import joblib
import pickle

joblib.dump(svm, "svm_model.pkl")

with open("class_names.pkl", "wb") as f:
    pickle.dump(list(train_gen.class_indices.keys()), f)
