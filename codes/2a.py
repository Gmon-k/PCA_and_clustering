
"""
2.Experiment with clustering and PCA on two structured data sets

A.Download and plot the example data

Download these two data sets: set A Download set Aand set B 
Download set B. Each data set has two features that are in the same 
units and ranges (no whitening necessary).


Submitted by : Gmon Kuzhiyanikkal
Date : 08 Feb 2023

"""
import matplotlib.pyplot as plt
import pandas as pd

# load csv file into a pandas DataFrame
df = pd.read_csv('clusterDataA-1.csv')

# plot x1 vs x2
plt.scatter(df['X1'], df['X2'])

# set x and y axis labels
plt.xlabel('x1_setA')
plt.ylabel('x2_setA')

# display the plot
plt.show()

df = pd.read_csv('clusterDataB-1.csv') #reading the csv file

# plot x1 vs x2
plt.scatter(df['X1'], df['X2'])

# set x and y axis labels
plt.xlabel('x1_setB')
plt.ylabel('x2_setB')

# display the plot
plt.show()
