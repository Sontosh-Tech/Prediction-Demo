import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

df=pd.read_csv('FuelConsumption.csv')

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

reg = LinearRegression()

x_ind = cdf.columns.values[0]
y_ind = cdf.columns.values[-1]
X = np.asanyarray(df[x_ind]).reshape(-1,1)
y = np.asanyarray(df[y_ind]).reshape(-1,1)

reg.fit(cdf[cdf.columns[:-1]],cdf[cdf.columns[-1]])

pickle.dump(reg, open('ML_model.pkl', 'wb'))
