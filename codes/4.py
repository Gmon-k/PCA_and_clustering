"""
4.Use K-Nearest Neighbor and PCA to classify activity from phone measurements

Use the sci-kit learn K-nearest neighbor algorithm to classify the UCI Activity 
Recognition data set Links to an external site.. The data set contains 10299 data 
points, each of which has 561 accelerometer measurements from a mobile phone

submitted by: Gmon kuzhiyanikkal
Date : 08 Feb 2023


"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the training data from text files
X_train = np.loadtxt("X_train.txt")
y_train = np.loadtxt("y_train.txt")

# Load the test data from text files
X_test = np.loadtxt("X_test.txt")
y_test = np.loadtxt("y_test.txt")

# Build KNN classifier using the raw data
knn_no_pca = KNeighborsClassifier(n_neighbors=5)
knn_no_pca.fit(X_train, y_train)
y_pred = knn_no_pca.predict(X_test)

# Evaluate the  KNN classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("KNN Classifier")
print('\n')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("------------------------")
print('\n')

# Performing PCA and keep only enough dimensions to explain 90% of the variance
pca = PCA(n_components=0.9)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Build KNN classifier using the PCA projected data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)

# Evaluate the PCA projected data KNN classifier
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, average='weighted')
recall_pca = recall_score(y_test, y_pred_pca, average='weighted')
f1_pca = f1_score(y_test, y_pred_pca, average='weighted')
print("\nPCA Projected Data KNN Classifier")
print('\n')
print("Accuracy:", accuracy_pca)
print("Precision:", precision_pca)
print("Recall:", recall_pca)
print("F1 Score:", f1_pca)
