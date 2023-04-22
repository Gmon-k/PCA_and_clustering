
"""
2.Experiment with clustering and PCA on two structured data sets

c)Compare using different numbers of clusters

Apply K-means clustering on data set A using K from 2 to 10. 
Plot the representation error for each value of K. You can use the sklearn 
transform function to obtain the distance of each point from its closest cluster mean.

submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataA-1.csv')

# array to store the representation error for each value of K
representation_error = []

# array to store the Rissannon Minimum Description Length for each value of K
rissannon_minimum_Length = []

# perform K-means clustering with K from 2 to 10
for k in range(2, 11):
    kmeans_clsuer = KMeans(n_clusters=k)
    kmeans_clsuer.fit(df)

    # calculate the representation error
    representation_error.append(kmeans_clsuer.inertia_)

    # calculate the Rissannon Minimum Description Length
    labels = kmeans_clsuer.labels_
    silhouette = silhouette_score(df, labels)
    rissannon_minimum_Length.append(-k * np.log(silhouette) - (len(df) - k) * np.log(1 - silhouette))

# plot the representation error
plt.plot(range(2, 11), representation_error)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Representation error')
plt.title('Representation error vs. Number of clusters')
plt.show()

# plot the Rissannon Minimum Description Length
plt.plot(range(2, 11), rissannon_minimum_Length)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Rissannon Minimum Description Length')
plt.title('Rissannon Minimum Description Length vs. Number of clusters')
plt.show()
