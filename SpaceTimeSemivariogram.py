
import skgstat as skg
from skgstat.plotting import backend
import numpy as np
import pandas as pd
from skgstat import Variogram
from plotly.subplots import make_subplots
import geopandas as gpd



df=pd.read_csv('E:\\Data\\Monicipolity\\137shp\\SadEMabarMounth.csv')

Mahallat = gpd.read_file('E:\\Data\\Monicipolity\\Mahalat\\Mahalat\\Mahalat.shp')

Mahallat.plot()

Cent=Mahallat.centroid



indexex=df.columns

sumColumn=df.sum()

indexex=indexex[1:]

sumColumn=sumColumn[1:]


sumColumn=sumColumn.astype(int)

indexex=indexex.astype(int)

Mahallat['CODE']=Mahallat['CODE'].astype(int)

test = np.vstack((sumColumn, indexex)).T

df2 = pd.DataFrame(data = test, columns = ['Sum','CODE'])

cshapes = Mahallat.merge(df2, on='CODE')


cshapes['X']=Cent.geometry.x

cshapes['Y']=Cent.geometry.y

coords = cshapes[['X','Y']].values


V = Variogram(coords, cshapes.Sum.values, normalize=False, maxlag=6000, n_lags=100)


fig = V.plot(show=False)



cshapes = cshapes.sort_values(by=['CODE'])

cshapes['X']=Cent.geometry.x

cshapes['Y']=Cent.geometry.y

coords = cshapes[['X','Y']].values

df = df.drop(df.columns[[0]], axis=1)

df_1=df.T

df_1 = pd.DataFrame(
    df_1.values,
    index = indexex.astype(int),
    columns=df_1.columns)

df_1 = df_1.sort_index(ascending=True)

df=df_1.T

valss=np.transpose(df.values)

valss=valss.astype(float)

STV = skg.SpaceTimeVariogram(coords, valss[:,0:20], x_lags=20, t_lags=20, model='product-sum')

fig = STV.plot()

fig=STV.plot(kind='surf')



fig = STV.plot(kind='contour')

fig = STV.plot(kind='contourf')

