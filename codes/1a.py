"""
Write a distance metric for your data set
Given your data, write a function that computes the distance between at least one data point (unknown point) 
and a set of data points (exemplar points).

submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023

"""


import pandas as pd
import numpy as np

#function used for the distance metric is eculidian metric
def euclidean_distance_metric(dataframe_exemplars, dataframe_unknowns_point):
    if dataframe_unknowns_point.ndim == 1:
        dataframe_unknowns_point = dataframe_unknowns_point.reshape(1, -1)

    #canceling the dependent attriubute from the csv file.

    exemplar_matrix = dataframe_exemplars.drop("wheatType", axis=1).values
    unknown_matrix = dataframe_unknowns_point.drop("wheatType", axis=1).values

    distances = np.sqrt(np.sum((exemplar_matrix - unknown_matrix[:, np.newaxis]) ** 2, axis=2))

    return distances


dataframe_exemplars_data = pd.read_csv("seed_data.csv") #passing the train_set
dataframe_unknowns_data = pd.read_csv("unknownpoint.csv") #passing an unknown point, here it is just one single point

distances = euclidean_distance_metric(dataframe_exemplars_data, dataframe_unknowns_data)
print(distances) #printing the distance metrics



