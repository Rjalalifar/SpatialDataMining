# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 07:57:45 2021

@author: ASUS
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split


from sqlalchemy import create_engine  
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

db_connection_url = "postgresql://postgres:12546@127.0.0.1:5432/SpatialDatabase"
con = create_engine(db_connection_url)  
sql = 'SELECT * FROM "NavahiFinsl"'

df = gpd.GeoDataFrame.from_postgis(sql, con)  
df.plot()



olivetti = fetch_olivetti_faces()

x = olivetti.images  # Train
y = olivetti.target  # Labels

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    random_state=42)

x_train = x_train.reshape((x_train.shape[0], x.shape[1] * x.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x.shape[1] * x.shape[2]))
x = x.reshape((x.shape[0]), x.shape[1] * x.shape[2])


from sklearn.decomposition import PCA
from matplotlib.pyplot import figure, get_cmap, colorbar, show

class_num = 40
sample_num = 10

pca = PCA(n_components=2).fit_transform(x)
idx_range = class_num * sample_num
fig = figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(1, 1, 1)
c_map = get_cmap(name='jet', lut=class_num)
scatter = ax.scatter(pca[:idx_range, 0], pca[:idx_range, 1], 
                     c=y[:idx_range],s=10, cmap=c_map)

ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")
ax.set_title("PCA projection of {} people".format(class_num))
colorbar(mappable=scatter)
show()


from matplotlib.pyplot import plot, xlabel, ylabel

pca2 = PCA().fit(x)
plot(pca2.explained_variance_, linewidth=2)
xlabel('Components')
ylabel('Explained Variaces')
show()


from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df2=df.iloc[: ,24:278]

x=df2.values

# x, _ = fetch_olivetti_faces(return_X_y=True)
pca2 = PCA().fit(x)
plt.plot(pca2.explained_variance_, linewidth=2)
plt.xlabel('Components')
plt.ylabel('Explained Variances')
plt.show()



pca = PCA(n_components=18).fit_transform(x)
