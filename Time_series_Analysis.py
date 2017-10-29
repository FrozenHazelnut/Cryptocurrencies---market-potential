#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:14:52 2017

@author: yixuanchai
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date
from io import StringIO
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6
%matplotlib inline
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    

#Get the data prepared
def parser(x):
    return datetime.datetime.strptime(x,'%d-%b-%y')
#Download data
series = pd.read_csv('Users/yixuanchai/Desktop/Price.csv',header=0,parse_dates=[0],index_col=0,squeeze=True,date_parser=parser)
print(series.head())
#Convert dtype and replace comma thus the price is manipulatable
series = pd.to_numeric(series.str.replace(',', '')) 
X = series.values
plt.plot(X)
#Check for stationarity
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=3,center=False).mean()
    rolstd = timeseries.rolling(window=3,center=False).std()
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistics', 'p-value', '#Lag Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(series)

#Test statistica is more than the 5% critical value; p value larger than 0.05: Not stationary
#Stationarie the time series is very necessary

series_log = np.log(series)
test_stationarity(series_log)

#Still non-stationary, remove trend and seasonality with decompostion

decomposition = seasonal_decompose(series)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(series_log[-84:],label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend[-84:],label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal[-84:],label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual[-84:],label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#Remove trend and seasonality w/ differencing
series_log_diff = series_log - series_log.shift()
series_log_diff.dropna(inplace=True)
plt.plot(series_log_diff)
test_stationarity(series_log_diff)

#Plot the ACF and PACF
lag_acf = acf(series_log_diff, nlags=10)
lag_pacf = pacf(series_log_diff,nlags=10,method='ols')
#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Build the ARIMA model with (p,d,q) order suggested by the function:
model = ARIMA(series_log, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)  
plt.plot(series_log_diff)  
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4F'% sum((results_ARIMA.fittedvalues-series_log_diff)**2))

#Predict Future price
Future_Price_log_diff = results_ARIMA.predict(84,120) #84-120 corresponds to 2017/01/01 TO 2019/12/01
original_price_log = series_log.iloc[0]
Future_Price_log = Future_Price.astype(float)
Future_Price_log = Future_Price.values + original_price_log
Future_Price = np.exp(Future_Price)

#Measure the variance between the data and the values predicted by the model
print(results_ARIMA.summary())
#Plot residual errors
residuals = pd.DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

#Result is good!

#Scale Predictions
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(series_log.iloc[0],index=series_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(series)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-series)**2)/len(series)))




#Build the ARIMA Model with train and test set (model selection)
size = int(len(series_log)*0.66)
train = series_log[0:size]
test = series_log[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat,obs))
error = mean_squared_error(test,predictions[len(predictions)-29:len(predictions)])
print('Test MSE: %.3f' % error)
predictions_series = pd.Series(predictions, index=test.index).values
predictions_series = predictions_series.astype(float)
exp_predictions = np.exp(predictions_series)


fig,ax = plt.subplots()
ax.set(title='Palm Oil Pricing',xlabel='Date',ylabel='Palm Oil Price')
ax.plot(series[-60:],'o',label='observed')
ax.plot(exp_predictions,'g',label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


plt.plot(exp_predictions,'g')
plt.plot(series,'o')





autocorrelation_plot(series)       
pyplot.show() #First 22-24 lags are positively autocorrelated, 
#Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())





#Cross Validation??
#Roll1
trainset1 = series.iloc[0:12]
testset1 = series.iloc[12:24]

#Roll2
trainset2 = series.iloc[0:24]
testset2 = series.iloc[24:36]

#Roll3
trainset3 = series.iloc[0:36]
testset3 = series.iloc[36:48]

#Roll4
trainset4 = series.iloc[0:48]
testset4 =series.iloc[48:60]

#Roll5
trainset5 = series.iloc[0:60]
testset5 = series.iloc[60:72]

#Roll6
trainset6 = series.iloc[0:72]
testset6 = series.iloc[72:]