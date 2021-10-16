#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:03:37 2020

@author: f1zhao_95
"""

import numpy as np # Python Library for numerical Funktion
import pandas as pd # For making Dataframe
import math
import time
import datetime


import sklearn # Python Library for linear and other models
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split # For train test split

import SimulationConfiguration as SimConfig

from sklearn.metrics import mean_squared_error
from math import sqrt

class PrognosePV():
    
    #def __init__(self,time_stamp = SimulationConfig.SIMULATION_BEGINNING)
    def run(self,timeindex):
        
        self.area = SimConfig.Area_PV
        self.efficiency = SimConfig.Efficiency_PV
        self.angle = SimConfig.Illumination_Angle
        PV = []
        if SimConfig.Forecast_PV == 'no' or SimConfig.Level == 1:
            df = pd.read_excel('./prognose/Eingangsdaten/200610_Prognose Globalstrahlung_fertig.xlsx')
            timeArray = time.strptime(timeindex, "%d.%m.%Y %H:%M")
            timestamp = time.mktime(timeArray)
            time_local = time.localtime(timestamp)
            dt = datetime.datetime(year = 2010, month=time_local.tm_mon,day = time_local.tm_mday)
            t = datetime.time(hour=time_local.tm_hour,minute=time_local.tm_min)
            df1 = df[df.iloc[:,0] == dt]
            df1 = df1[df1.iloc[:,1] == t]
            a = df1.index[0]
            for i in range(a,a+97):
                GS = df.iloc[i,8]
                cos = math.cos(math.pi*self.angle/180)
                PV_power = GS*cos*0.01*self.efficiency*self.area*0.001
                PV.append(PV_power)
            df_save = pd.DataFrame(data=PV)
            df_save.to_csv('./prognose/Prognose_PVPower.csv',header = False, index = False)
        else:
            # Making a Dataframe for independent variables
            #A variable for prediction n days out into  Future
            self.df = pd.read_csv('./prognose/Eingangsdaten/Globalstrahlung.csv', sep= ';', index_col=0, parse_dates=True)
    
            self.row = np.where(self.df.index==timeindex)[0]
            self.row = int(self.row)
    
            self.forecast_out = self.row
    
            #create another column shifted n unites up
            self.df['Prognose']= self.df[['Globalstrahlung']].shift(-self.forecast_out)
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
            self.svr_rbf = SVR(kernel='rbf', C=1e3 , gamma=10e2)
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
            
            #Print support vector regressor model for the Prognose for n days
            self.svm_predection = self.svr_rbf.predict(self.x_forecast)
            
            self.cos = math.cos(math.pi*self.angle/180)
            
            self.PV_power = self.svm_predection*self.cos*0.01*self.efficiency*self.area*0.001 #kW
            #print(svm_predection)
            #self.t = pd.date_range(start=(self.df.index[len(self.df)-1]) , periods=24+1, freq='15min')
            #self.t1= pd.Series(self.t)
            #self.result2 = pd.Series(self.lr_predection)
            #self.result3=pd.DataFrame(columns = ['Date', 'Forecasting'])
            #self.result3['Date']=self.t1
            #self.result3['Forecasting']=self.result2
            #self.result3= self.result3[1:]
            #return self.result3

            np.savetxt('./prognose/Prognose_PVPower.csv', self.PV_power, delimiter=';')

    def parameter(self):
        self.df = pd.read_csv('./prognose/Eingangsdaten/Globalstrahlung.csv', sep=';', index_col=0, parse_dates=True)

        self.row = np.where(self.df.index == timeindex)[0]
        self.row = int(self.row)

        self.forecast_out = self.row

        # create another column shifted n unites up
        self.df['Prognose'] = self.df[['Globalstrahlung']].shift(-self.forecast_out)
        self.df.tail( )

        # Create the independant data set (X)
        # convert data frame into numpy array
        self.X = np.array(self.df.drop(['Prognose'], 1))
        # remove the last n rows
        self.X = self.X[:-self.forecast_out]
        # print(X)

        # create the dependent data set Y
        self.y = np.array(self.df['Prognose'])
        # Get all of the y values except the last n rows
        self.y = self.y[:-self.forecast_out]
        # print(y)

        # split the data into 80%training and 20% testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 5, 10, 10e2, 10e4], 'gamma': {0.1, 0.5, 1, 10, 10}}

        clf = RandomizedSearchCV(SVR( ), parameters)
        clf = GridSearchCV(SVR( ), parameters)
        clf.fit(x_train, y_train)
        print('score', clf.score(x_test, y_test))
        print(clf.best_params_)


        # create and train the support vector machine (Regressor)
        self.svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        self.svr_rbf.fit(self.x_train, self.y_train)

        # Testing Model
        # best possible score is 1
        self.svm_confidence = self.svr_rbf.score(self.x_test, self.y_test)
        # print('svm confidence:', svm_confidence)


if __name__ == '__main__':
    test = PrognosePV()
    test.run(SimConfig.timeindex)

    daten = pd.read_csv('./prognose/Eingangsdaten/Globalstrahlung.csv', sep= ';', index_col=0, parse_dates=True)

    y_truth = daten[test.row:test.row+97]

    rmse = sqrt(mean_squared_error(y_truth, test.svm_predection))
    print('RMSE SVM: ', rmse)
    

