import skgstat as skg
from skgstat.plotting import backend
import numpy as np
import pandas as pd
import json
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import geopandas as gpd
import pywt
import pywt.data


df=pd.read_csv(r'E:\projects\PythonDatamining\Gold Price.csv')


y = df.values[0:,1]


plt.plot(y)

w = pywt.Wavelet('sym5')
plt.plot(w.dec_lo)

coeffs = pywt.wavedec(y, w, level=6)


plt.plot(pywt.waverec(coeffs, w))
plt.plot(pywt.waverec(coeffs[:-1] + [None] * 1, w))
plt.plot(pywt.waverec(coeffs[:-2] + [None] * 2, w))
plt.plot(pywt.waverec(coeffs[:-3] + [None] * 3, w))
plt.plot(pywt.waverec(coeffs[:-4] + [None] * 4, w))
plt.plot(pywt.waverec(coeffs[:-5] + [None] * 5, w))
plt.plot(pywt.waverec(coeffs[:-6] + [None] * 6, w))


plt.plot(coeffs[1]); plt.legend(['Lvl 1 detail'])
plt.plot(coeffs[2]); plt.legend(['Lvl 2 detail'])
plt.plot(coeffs[3]); plt.legend(['Lvl 3 detail'])
plt.plot(coeffs[4]); plt.legend(['Lvl 4 detail'])
plt.plot(coeffs[5]); plt.legend(['Lvl 5 detail'])
plt.plot(coeffs[6]); plt.legend(['Lvl 6 detail'])
