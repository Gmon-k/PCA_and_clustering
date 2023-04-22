
"""
Extensions:

as a part of extension, i tired to compare two cluster quality metric , that is Rissannon Minimum Description Length
and WithinCluster Sum of Squared Errors. In addition to that, i have tried to plot
the number of cluster vs quality metric graph for both of the cases.

submitted by: Gmon kuzhiyanikkal
Date : 08 Feb 2023
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#function definition of the within cluster sum
def Within_Cluster_Sum(X, labels, centroids):
    n = X.shape[0]
    k = centroids.shape[0]
    distances = np.zeros(n)
    for i in range(k):
        mask = (labels == i)
        distances[mask] = np.sum((X[mask] - centroids[i])**2, axis=1)
    return np.sum(distances)

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataA-1.csv')

# store the representation error for each value of K
representation_error = []

# store the Rissannon Minimum Description Length for each value of K
rissannon_mimimum_length = []

# store the Within-Cluster-Sum-of-Squared-Errors for each value of K
within_cluster_sum = []

# perform K-means clustering with K from 2 to 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)

    # calculate the Rissannon Minimum Description Length
    labels = kmeans.labels_
    silhouette = silhouette_score(df, labels)
    rissannon_mimimum_length.append(-k * np.log(silhouette) - (len(df) - k) * np.log(1 - silhouette))

    # calculate the Within-Cluster-Sum-of-Squared-Errors
    centroids = kmeans.cluster_centers_
    within_cluster_sum.append(Within_Cluster_Sum(df.values, labels, centroids))

# plot the Rissannon Minimum Description Length
plt.plot(range(2, 11), rissannon_mimimum_length)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Rissannon Minimum Description Length')
plt.title('Rissannon Minimum Description Length vs. Number of clusters')
plt.show()

# plot the Within-Cluster-Sum-of-Squared-Errors
plt.plot(range(2, 11), within_cluster_sum)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster-Sum-of-Squared-Errors')
plt.title('Within-Cluster-Sum-of-Squared-Errors vs. Number of clusters')
plt.show()
