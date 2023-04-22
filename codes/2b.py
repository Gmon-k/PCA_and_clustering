
"""
2.Experiment with clustering and PCA on two structured data sets

B.Cluster the example data

Use the sklearn k-means clustering algorithm to cluster both of the data sets. 
Generate a plot for each data set using K=6, using color to indicate the cluster 
ID for each point.

submitted by: Gmon kuzhiyanikkal
Date : 08 Feb 2023

"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataA-1.csv')

# perform k-means clustering with k=6
kmeans = KMeans(n_clusters=6)
kmeans.fit(df)
labels_kmeans = kmeans.labels_

# perform hierarchical clustering with k=6
agg_clustering = AgglomerativeClustering(n_clusters=6)
agg_clustering.fit(df)
labels_agg = agg_clustering.labels_

# create a scatter plot with color indicating the cluster ID for each point
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.scatter(df['X1'], df['X2'], c=labels_kmeans, cmap='rainbow')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('K-Means Clustering')

ax2.scatter(df['X1'], df['X2'], c=labels_agg, cmap='rainbow')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Hierarchical Clustering')

plt.show()

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataB-1.csv')

# perform k-means clustering with k=6
kmeans = KMeans(n_clusters=6)
kmeans.fit(df)
labels_kmeans = kmeans.labels_

# perform hierarchical clustering with k=6
agg_clustering = AgglomerativeClustering(n_clusters=6)
agg_clustering.fit(df)
labels_agg = agg_clustering.labels_

# create a scatter plot with color indicating the cluster ID for each point
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.scatter(df['X1'], df['X2'], c=labels_kmeans, cmap='rainbow')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('K-Means Clustering')

ax2.scatter(df['X1'], df['X2'], c=labels_agg, cmap='rainbow')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Hierarchical Clustering')

plt.show()

