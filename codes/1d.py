
"""
D: Experiment with your classifier
Make at least one modification to your classifier to try and improve its performance (speed or accuracy) and re-evaluate it. Some possibilities include: 
(1) try a different distance metric or a modification of the one you wrote

Submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023

"""

import pandas as pd
import numpy as np


#function defeintion of the distance metrics(used cosine similarity distance metric)
def cosine_distance(dataframe_train, dataframe_test):
    #dropping of the dependent variable  
    exemplar_matrix = dataframe_train.drop("wheatType", axis=1).values
    unknown_matrix = dataframe_test.drop("wheatType", axis=1).values
    #calculation the dot product
    dot_product = np.dot(exemplar_matrix, unknown_matrix.T)
    exemplar_norm = np.linalg.norm(exemplar_matrix, axis=1)
    unknown_norm = np.linalg.norm(unknown_matrix, axis=1)
    pairwise_products = np.outer(exemplar_norm, unknown_norm)
    cosine_similarity = dot_product / pairwise_products
    return 1 - cosine_similarity



#function for implmenting the nearest_neigbour
def nearest_neighbor_classification(dataframe_train, dataframe_test):
    distances = cosine_distance(dataframe_train, dataframe_test)
    nearest_neighbors = dataframe_train.iloc[np.argmin(distances, axis=0)]
    #getting the predicted classes
    predicted_classes = nearest_neighbors["wheatType"].values
    #getting the error terms
    error = np.min(distances, axis=0)
    return predicted_classes, error


#function to calculate the accuracy of the predicted class and true_classes(atucal test data)
def accuracy(true_classes, predicted_classes):
    correct = np.sum(true_classes == predicted_classes)
    accuracy = correct / len(true_classes)
    return accuracy

dataframe_train_set = pd.read_csv("seed_data.csv") #train_data_set is passed
dataframe_test_set = pd.read_csv("test_set.csv")  #test_data_set is passed

predicted_classes, error_terms = nearest_neighbor_classification(dataframe_train_set, dataframe_test_set)
true_classes = dataframe_test_set["wheatType"].values


#printing the accuracy
print("---accuracy using cosine similarity------")
print('\n')
accuracy = accuracy(true_classes, predicted_classes)
print("Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import confusion_matrix

#calcualting the confusin matrix, as part of 1c(task from the project)
def calculate_confusion_matrix(true_classes, predicted_classes):
    predicted_classes = predicted_classes[:len(true_classes)]
    return confusion_matrix(true_classes, predicted_classes)
print('\n')
conf_matrix = calculate_confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)#prining the confusion matrix



