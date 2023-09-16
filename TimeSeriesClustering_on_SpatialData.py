import skgstat as skg
from skgstat.plotting import backend
import numpy 
import pandas as pd
import json
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import geopandas as gpd

from tslearn.clustering import KShape

from tslearn.preprocessing import TimeSeriesScalerMeanVariance


df=pd.read_csv('E:\\Data\\Monicipolity\\137shp\\SadEMabarMounth.csv')

Mahallat = gpd.read_file('E:\\Data\\Monicipolity\\Mahalat\\Mahalat\\Mahalat.shp')

Mahallat.plot()

indexex=df.columns

indexex=indexex[1:]

df=numpy.transpose(df.values[:,1:])

X_train = TimeSeriesScalerMeanVariance().fit_transform(df)
sz = X_train.shape[1]

ks = KShape(n_clusters=4, verbose=True)
y_pred = ks.fit_predict(X_train)


plt.figure()
for yi in range(4):
    plt.subplot(4, 1, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()

y_pred=y_pred.astype(int)

indexex=indexex.astype(int)

Mahallat['CODE']=Mahallat['CODE'].astype(int)

test = numpy.vstack((y_pred, indexex)).T

df2 = pd.DataFrame(data = test, columns = ['pred','CODE'])

cshapes = Mahallat.merge(df2, on='CODE')



cshapes.plot(column='pred')
plt.show()
