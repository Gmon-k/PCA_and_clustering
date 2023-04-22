
"""
B.Implement Nearest Neighbor classification for your test set
C.Evaluate your classifier

both are done in a single python file

submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023
"""



import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

#function defeintion of the distance metrics
def euclidean_distance_metric(dataframe_exemplars, dataframe_unknowns_point):
    if dataframe_unknowns_point.ndim == 1:
        dataframe_unknowns_point = dataframe_unknowns_point.reshape(1, -1) 
    #dropping of the dependent variable  
    exemplar_matrix = dataframe_exemplars.drop("wheatType", axis=1).values
    unknown_matrix = dataframe_unknowns_point.drop("wheatType", axis=1).values
    distances = np.sqrt(np.sum((exemplar_matrix - unknown_matrix[:, np.newaxis])**2, axis=2))
    return distances  # returning the distance metric


#function for implmenting the nearest_neigbour
def nearest_neighbor_classification(dataframe_exemplars, dataframe_unknowns):
    distances = euclidean_distance_metric(dataframe_exemplars, dataframe_unknowns)
    nearest_neighbors = dataframe_exemplars.iloc[np.argmin(distances, axis=0)]
    #getting the predicted classes
    predicted_classes = nearest_neighbors["wheatType"].values
    #getting the error terms
    error = np.min(distances, axis=0)   
    return predicted_classes, error



#function to calculate the accuracy of the predicted class and true_classes(atucal test data)
def accuracy(true_classes, predicted_classes):
    predicted_classes = predicted_classes.astype(str)
    true_classes = true_classes.astype(str)
    i,correct = 0,0
    while(i<len(true_classes)):
        if(true_classes[i]==predicted_classes[i]):
            correct = correct+1
        i=i+1
    accuracy = correct / len(true_classes)
    return accuracy



dataframe_train_set = pd.read_csv("seed_data.csv")  #train_data_set is passed
dataframe_test_set = pd.read_csv("test_set.csv")    #test_data_set is passed

predicted_classes, error_terms = nearest_neighbor_classification(dataframe_train_set, dataframe_test_set)
#printing the error terms.
print("--------error terms------------")
print('\n')
print(error_terms)

true_classes = dataframe_test_set["wheatType"].values
acc = accuracy(true_classes, predicted_classes)

#printing the accuracy
print('\n')
print("Accuracy of the Model using the Eculcidian distance")
print("Accuracy: {:.2f}%".format(acc * 100))


#calcualting the confusin matrix, as part of 1c(task from the project)
print('\n')
def calculate_confusion_matrix(true_classes, predicted_classes):
    predicted_classes = predicted_classes[:len(true_classes)]
    return confusion_matrix(true_classes, predicted_classes)

conf_matrix = calculate_confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", conf_matrix) #prining the confusion matrix

