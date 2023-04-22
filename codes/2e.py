"""
2.Experiment with clustering and PCA on two structured data sets

E)Recluster using the projected data

Repeat the clustering process with the same two algorithms 
(K-means plus one of your choice) on the PCA-projected data. Plot the clusters 
and discuss any difference.

submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023

"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataA-1.csv')

# perform PCA dimensionality reduction
pca = PCA(n_components=2)
pca_projected = pca.fit_transform(df)

# perform k-means clustering with k=6
kmeans_cluster = KMeans(n_clusters=6)
kmeans_cluster.fit(pca_projected)
kmeans_label_cluster = kmeans_cluster.labels_

# perform hierarchical clustering with k=6
agglomerative_clustering = AgglomerativeClustering(n_clusters=6)
agglomerative_clustering.fit(pca_projected)
labels_agg = agglomerative_clustering.labels_

# create a scatter plot with color indicating the cluster ID for each point
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.scatter(pca_projected[:, 0], pca_projected[:, 1], c=kmeans_label_cluster, cmap='rainbow')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_title('K-Means Clustering')

ax2.scatter(pca_projected[:, 0], pca_projected[:, 1], c=labels_agg, cmap='rainbow')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_title('Hierarchical Clustering')

plt.show()

#---------
#perfrom the same operation for the second data set B
#---------


# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataB-1.csv')

# perform PCA dimensionality reduction
pca = PCA(n_components=2)
pca_projected = pca.fit_transform(df)

# perform k-means clustering with k=6
kmeans_cluster = KMeans(n_clusters=6)
kmeans_cluster.fit(pca_projected)
kmeans_label_cluster = kmeans_cluster.labels_

# perform hierarchical clustering with k=6
agglomerative_clustering = AgglomerativeClustering(n_clusters=6)
agglomerative_clustering.fit(pca_projected)
labels_agg = agglomerative_clustering.labels_

# create a scatter plot with 6 different color indicating the cluster ID for each point
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.scatter(pca_projected[:, 0], pca_projected[:, 1], c=kmeans_label_cluster, cmap='rainbow')
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_title('K-Means Clustering')

ax2.scatter(pca_projected[:, 0], pca_projected[:, 1], c=labels_agg, cmap='rainbow')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_title('Hierarchical Clustering')

plt.show()
