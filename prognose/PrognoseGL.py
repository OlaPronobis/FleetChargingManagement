#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:20:01 2020

@author: f1zhao_95
"""

import numpy as np # Python Library for numerical Funktion
import pandas as pd # For making Dataframe
import time
import datetime
import matplotlib.pyplot as plt 
#get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn # Python Library for linear and other models
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split # For train test split

import SimulationConfiguration as SimConfig


class PrognoseGL():
    
    #def __init__(self,time_stamp = SimulationConfig.SIMULATION_BEGINNING)
    def run(self,timeindex):
        GL = []
        if SimConfig.Forecast_building_load == 'no' or SimConfig.Level == 1:
            df = pd.read_excel('./prognose/Eingangsdaten/200606_SLP-G3_Niedersachsenbearbeitet.xlsm')
            timeArray = time.strptime(timeindex, "%d.%m.%Y %H:%M")
            timestamp = time.mktime(timeArray)
            time_local = time.localtime(timestamp)
            dt = datetime.datetime(year = 2018, month=time_local.tm_mon,day = time_local.tm_mday)
            t = datetime.time(hour=time_local.tm_hour,minute=time_local.tm_min)
            df1 = df[df.iloc[:,3] == dt]
            df1 = df1[df1.iloc[:,4] == t]
            a = df1.index[0]
            for i in range(a,a+97):
                PW = df.iloc[i,7]
                GL.append(PW)
            df_save = pd.DataFrame(data=GL)
            df_save.to_csv('./prognose/Prognose_Gebaeudelast.csv',header = False, index = False)
        else:
            # Making a Dataframe for independent variables
               
            #A variable for prediction n days out into  Future
            self.df = pd.read_csv('./prognose/Eingangsdaten/Gebaeudelast.csv', sep= ';', index_col=0, parse_dates=True)
            
            self.row = np.where(self.df.index==timeindex)[0]
            self.row = int(self.row)
    
            self.forecast_out = self.row
            
            #create another column shifted n unites up
            self.df['Prognose']= self.df[['kWh']].shift(-self.forecast_out)
            self.df.tail()
            
            #Create the independant data set (X)
            #convert data frame into numpy array
            self.X = np.array(self.df.drop(['Prognose'],1))
            #remove the last n rows
            self.X = self.X[:-self.forecast_out]
            #print(X)
            
            
            #create the dependent data set Y
            self.y = np.array(self.df['Prognose'])
            # Get all of the y values except the last n rows
            self.y = self.y[:-self.forecast_out]
            #print(y)
            
            
            #split the data into 80%training and 20% testing
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
            
            #create and train the support vector machine (Regressor)
            self.svr_rbf = SVR(kernel='rbf', C=1e3 , gamma=0.1)
            self.svr_rbf.fit(self.x_train, self.y_train)
            
            #Testing Model
            #best possible score is 1
            self.svm_confidence = self.svr_rbf.score(self.x_test, self.y_test)
            #print('svm confidence:', svm_confidence)
            
            
            #create and train the Linear Regression Model
            self.lr = LinearRegression()
            self.lr.fit(self.x_train, self.y_train)
            
            #Testing Model
            #best possible score is 1
            self.lr_confidence = self.lr.score(self.x_test, self.y_test)
            #print('lr confidence:', lr_confidence)
            
            #set x_forecast
            self.x_forecast = np.array(self.df.drop(['Prognose'],1))[-97:]
            #print(x_forecast)
            
            #Print linear Regression Model for the Prognose for n days
            self.lr_predection = self.lr.predict(self.x_forecast)
            #print(lr_predection)
            
            #Print support vector regressor model for the Prognose for n days
            self.svm_predection = self.svr_rbf.predict(self.x_forecast)
            #print(svm_predection)
            #self.t = pd.date_range(start=(self.df.index[len(self.df)-1]) , periods=24+1, freq='15min')
            #self.t1= pd.Series(self.t)
            #self.result2 = pd.Series(self.lr_predection)
            #self.result3=pd.DataFrame(columns = ['Date', 'Forecasting'])
            #self.result3['Date']=self.t1
            #self.result3['Forecasting']=self.result2
            #self.result3= self.result3[1:]
            #return self.result3
            np.savetxt('./prognose/Prognose_Gebaeudelast.csv', self.svm_predection, delimiter=';')
        
    
