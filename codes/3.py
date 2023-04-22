"""

3.Apply K-Means Clustering to your data set

Use K-means to cluster your own data for a range of K you think is appropriate 
(keep the numbers small). Are there natural clusters in your data set? If you 
apply the cluster quality metric you implemented earlier, what does it tell you 
about the best K for your data set?

submitted by: Gmon kuzhiyanikkal
Date : 08 Feb 2023

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load csv file into a pandas DataFrame
df = pd.read_csv('seed_data.csv')

# store the representation error for each value of K
representation_error = []

# store the Rissannon Minimum Description Length for each value of K
rissannon_minimum = []

# perform K-means clustering with K from 2 to 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)

    # calculate the representation error
    representation_error.append(kmeans.inertia_)

    # calculate the Rissannon Minimum Description Length
    labels = kmeans.labels_
    silhouette = silhouette_score(df, labels)
    rissannon_minimum.append(-k * np.log(silhouette) - (len(df) - k) * np.log(1 - silhouette))

# plot the representation error
plt.plot(range(2, 11), representation_error)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Representation error')
plt.title('Representation error vs Number of clusters')
plt.show()

# plot the Rissannon Minimum Description Length
plt.plot(range(2, 11), rissannon_minimum)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Rissannon Minimum Description Length')
plt.title('Rissannon Minimum Description Length vs Number of clusters')
plt.show()