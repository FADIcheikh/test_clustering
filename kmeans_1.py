# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage



df = pd.read_table("D:\Data_Minig\seacnce3_clustering\cars_origin.txt",sep ='\t',header = 0)
df_new = df.drop(['origin'],axis=1)
print df_new.columns
############Kmeans##############
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_new)
print(kmeans.cluster_centers_)
print(kmeans.labels_)


d={'cluster':kmeans.labels_,'origin':df['origin']}
df_new_new = pd.DataFrame(data=d)
print df_new_new.columns
crosstable = pd.crosstab(pd.cut(df_new_new['cluster'], bins = 3, include_lowest=True,precision=3), df_new_new['origin'])
print crosstable

"""
plt.scatter(df_new['acceleration'], df_new['horsepower'], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('acceleration')
plt.ylabel('horsepower')

plt.show()
"""

#####################CAH##########################
link =linkage(df_new,method='ward',metric='euclidean')
plt.title("CAH")
dendrogram(link)
plt.show()
