"""
2.Experiment with clustering and PCA on two structured data sets

d)Apply PCA to the example data

Use your PCA code from project 1 to project both data sets onto their respective 
two eigenvectors. Generate a plot of the data with the direction of the first eigenvector 
shown as an arrow on the plot

submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataA-1.csv')

# perform PCA on the data
pca = PCA(n_components=2)
pca.fit(df)

# project the data onto the two eigenvectors
dataframe_pca = pca.transform(df)

# plotting the data
plt.scatter(dataframe_pca[:, 0], dataframe_pca[:, 1], c='b')

# plotting the first eigenvector
mean = np.mean(dataframe_pca, axis=0)
eigenvector1 = pca.components_[0]
x = np.array([mean[0], mean[0] + eigenvector1[0]])
y = np.array([mean[1], mean[1] + eigenvector1[1]])
plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], head_width=0.5, head_length=2, fc='k', ec='k')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data set A projected onto its two eigenvectors')
plt.show()


#---------
#perfrom the same operation for the second data set B
#---------

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataB-1.csv')

# perform PCA on the data
pca = PCA(n_components=2)
pca.fit(df)

# project the data onto the two eigenvectors
dataframe_pca = pca.transform(df)

# plotting the data
plt.scatter(dataframe_pca[:, 0], dataframe_pca[:, 1], c='b')

# plotting the first eigenvector
mean = np.mean(dataframe_pca, axis=0)
eigenvector1 = pca.components_[0]
x = np.array([mean[0], mean[0] + eigenvector1[0]])
y = np.array([mean[1], mean[1] + eigenvector1[1]])
plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], head_width=0.5, head_length=2, fc='k', ec='k')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data set B projected onto its two eigenvectors')
plt.show()
