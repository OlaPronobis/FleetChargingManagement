#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:50:07 2020

@author: f1zhao_95
"""


import numpy as np # Python Library for numerical Funktion
import pandas as pd # For making Dataframe
from pandas import datetime
import matplotlib.pyplot as plt 
#get_ipython().run_line_magic('matplotlib', 'inline')

import time

import sklearn # Python Library for linear and other models
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split # For train test split

import SimulationConfiguration as SimConfig


class PrognoseBK():
    
    #def __init__(self,time_stamp = SimulationConfig.SIMULATION_BEGINNING)
    def run(self,timeindex):
        # Making a Dataframe for independent variables
        timer = time.time( )
        
        #A variable for prediction n days out into  Future
        self.df = pd.read_csv('./prognose/Eingangsdaten/Energiequelle.csv', sep= ';', index_col=0)#, parse_dates=True)
        self.df.index = pd.to_datetime(self.df.index, format='%d.%m.%Y %H:%M')
        pd.set_option('display.max_rows', None)
        print(self.df)

        self.row = np.where(self.df.index==timeindex)[0]
        self.row = int(self.row)

        self.forecast_out = self.row
        print('forcasteout' , self.forecast_out) # ab diesem Datum/Zeit sind die Werte unbekannt
        
        #create another column shifted n unites up
        self.df['Prognose']= self.df[['Braunkohle']]#.shift(-self.forecast_out)
        #self.df.tail()

        #Create the independant data set (X)
        #convert data frame into numpy array
       # self.X = np.array(self.df.drop(['Prognose'],1))
        #remove the last n rows
       # self.X = self.X[:-self.forecast_out]
        #print(X)

        self.X_full = []
        i=0
        for data in self.df.index: # hier sind die x werte nur die daten
            date_int = data.timestamp()
            if len(self.X_full)> 2 and date_int-900 != self.X_full[-1][0]:
                print('falsch')
                i +=1
                print(i)
                print('jetztige zeot' ,data)
                print('vorherige', datetime.fromtimestamp(self.X_full[-1][0]))
            self.X_full.append([date_int])

           # print(date_int)
        print(self.X_full)
        self.X = self.X_full[:self.forecast_out]

        # new time depending on 15min interval
       # self.X_full = []
       # i=0
       # for data in self.df.index:  # hier sind die x werte nur die daten

       #     self.X_full.append([i])
       #     i += 1
       # print(self.X_full)
       # print(i)
       # self.X = self.X_full[:self.forecast_out]
       # print(self.X)

        #create the dependent data set Y
        self.y = np.array(self.df['Prognose'])
        # Get all of the y values except the last n rows
        self.Y_werte = self.y
        self.y = self.y[:self.forecast_out]
        #print(y)
        print('shapex ', len(self.X))#[0].shape())
        print('shapey ', len(self.y))#shape( ))

         #plt.scatter(self.x_test, self.y_test, color='red')
       # plt.plot(self.X, self.y, color='blue')
       # plt.title('plot 2 ')
       # plt.show( )
        
        
        #split the data into 80%training and 20% testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        
        #create and train the support vector machine (Regressor)
        self.svr_rbf = SVR(kernel='rbf', C=1e3 , gamma=0.1)

        print('svr_rbf anfang', time.time( ) - timer)
       # self.svr_rbf.fit(self.x_train, self.y_train)
        print('svr_rbf ende', time.time( ) - timer)
        
        #Testing Model
        #best possible score is 1
       # self.svm_confidence = self.svr_rbf.score(self.x_test, self.y_test)
       # print('svm confidence:', self.svm_confidence)
        
        
        #create and train the Linear Regression Model
        self.lr = LinearRegression()
        self.lr.fit(self.x_train, self.y_train)
        
        #Testing Model
        #best possible score is 1
        self.lr_confidence = self.lr.score(self.x_test, self.y_test)
        print('lr confidence:', self.lr_confidence)
        
        #set x_forecast
        #self.x_forecast = np.array(self.df.drop(['Prognose'],1))[-97:]
        #print(x_forecast)
        #self.x_forecast = self.X_full[-97:] #woher 97?? wieso nicht ab forcast out und dann zb. 3h/ prognose intervall?
        intervall =  24*4 # interalll in h * 4 -> 15min sind eine zeile
        self.x_forecast = self.X_full[self.forecast_out:self.forecast_out+intervall]

        
        #Print linear Regression Model for the Prognose for n days
        self.lr_predection = self.lr.predict(self.x_forecast)
        print(self.lr_predection)
        
        #Print support vector regressor model for the Prognose for n days
        #self.svm_predection = self.svr_rbf.predict(self.x_forecast)
        #print(self.svm_predection)
        #self.t = pd.date_range(start=(self.df.index[len(self.df)-1]) , periods=24+1, freq='15min')
        #self.t1= pd.Series(self.t)
        #self.result2 = pd.Series(self.lr_predection)
        #self.result3=pd.DataFrame(columns = ['Date', 'Forecasting'])
        #self.result3['Date']=self.t1
        #self.result3['Forecasting']=self.result2
        #self.result3= self.result3[1:]
        #return self.result3

        #plt.scatter(self.x_test, self.y_test, color='red')
        #plt.plot(self.X_full[-97:], self.svm_predection, color='blue')
        #plt.title('plot svm ')
        #plt.show( )

        y_werte = np.concatenate((self.lr.predict(self.X), self.lr_predection))
        print(y_werte)
        print(len(y_werte))
        x_achse = np.concatenate((self.X, self.x_forecast))

       # plt.scatter(self.x_test, self.y_test, color='red')
       # plt.plot(x_achse,y_werte, color='blue')
       # plt.plot(self.x_forecast, self.svm_predection, color = 'green')
      #  plt.plot(self.x_forecast, self.lr_predection, color='yellow')
      #  plt.title('plot lr test')
        #plt.show( )


        y_wahre = self.Y_werte[self.forecast_out:self.forecast_out+intervall]
        plt.plot(self.x_forecast, self.lr_predection, color='yellow')
      #  plt.plot(self.x_forecast, self.svm_predection, color='blue')
        plt.title('plot lr, svm test')
        plt.plot(self.x_forecast, y_wahre, color='green')
        plt.show()
        # plt.scatter(self.x_train, self.y_train, color='red')
        # plt.plot(x_achse, y_werte, color='blue')
        # plt.title('plot lr training ')
        # plt.show( )
        #
        # y_werte = np.concatenate((self.svr_rbf.predict(self.X), self.svm_predection))
        # plt.scatter(self.x_train, self.y_train, color='red')
        # plt.plot(x_achse, y_werte, color='blue')
        # plt.title('plot lr training ')
        # plt.show( )
        # plt.scatter(self.x_test, self.y_test, color='red')
        # plt.plot(x_achse, y_werte, color='blue')
        # plt.title('plot svm test ')
        # plt.show( )

        #print('15.04.2019 predict svm', self.svr_rbf.predict([[datetime(2019,4,15,0,0).timestamp()]]))
        print('15.04.2019 predict lr', self.lr.predict([[datetime(2019, 4, 15, 0, 0).timestamp( )]]))

        np.savetxt('./prognose/Prognose_Braunkohle.csv', self.lr_predection, delimiter=';')


def runs(self, timeindex):
    # Making a Dataframe for independent variables

    timer = time.time( )

    # A variable for prediction n days out into  Future
    # self.df = pd.read_csv('./prognose/Eingangsdaten/Energiequelle.csv', sep= ';', index_col=0, parse_dates=True)
    self.df = pd.read_csv('Eingangsdaten/Energiequelle.csv', sep=';', index_col=0, parse_dates=True)

    self.row = np.where(self.df.index == timeindex)[0]
    self.row = int(self.row)

    self.forecast_out = self.row

    # create another column shifted n unites up
    self.df['Prognose'] = self.df[['Braunkohle']].shift(-self.forecast_out)
    self.df.tail( )
    print(self.df)
    print(self.df['Prognose'])
    print('df tail', time.time( ) - timer)

    # Create the independant data set (X)
    # convert data frame into numpy array
    # data = self.df['Braunkohle']
    data = self.df.drop(['Prognose', 'Biomasse', 'Wasserkraft', 'WindOnshore', 'Kernenergie', 'Steinkohle'
                            , 'SonstigeKonventionelle', 'Erdgas', 'GesamteerzeugteEnergie', 'EEEnergie[MWh]',
                         'Anteil'], axis=1)
    self.X = np.array(data)
    # self.X = np.array(self.df.drop(['Prognose'],axis=1)) #droppt die reihen/werte ??
    # array arbeitet ohne die prognose werte?
    # array aufbau:
    # 11 coumln: daten, biomasse, wssaer ,... ,eeee, anteil
    print(data)

    print('self X')
    # remove the last n rows
    self.X = self.X[:-self.forecast_out]
    # print(X)
    print(self.X)
    print('np array x', time.time( ) - timer)

    # create the dependent data set Y
    self.y = np.array(self.df['Prognose'])  # 1d array/liste mit den werten bis zum zeitplunkt
    # NUR WERTE bis zum zeitpunkt x (ohne zeitpunkten)
    print('####')
    print(self.df['Prognose'])

    # Get all of the y values except the last n rows
    self.y = self.y[:-self.forecast_out]
    # print(y)
    print('np arrey y', time.time( ) - timer)

    print(self.y)

    # plt.scatter(self.df.index.values, self.df['Braunkohle'], color='red')
    # plt.plot(self.x_train, self.svm_predection, color='blue')
    # plt.title('plot 0')
    # plt.show( )
    # plt.scatter(self.X, self.X, color='red')
    # plt.plot(self.x_train, self.svm_predection, color='blue')
    # plt.title('plot 0')
    # plt.show( )

    # split the data into 80%training and 20% testing
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
    print('train', time.time( ) - timer)
    # create and train the support vector machine (Regressor)
    print(self.x_train)
    print('####')
    print(self.y_train)
    self.svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    print('SVR', time.time( ) - timer)
    self.svr_rbf.fit(self.x_train, self.y_train)  # ------------
    print('svr_rbf ende', time.time( ) - timer)
    # Testing Model
    # best possible score is 1
    self.svm_confidence = self.svr_rbf.score(self.x_test, self.y_test)
    print('svm confidence:', self.svm_confidence)

    print('svm', time.time( ) - timer)

    # create and train the Linear Regression Model
    self.lr = LinearRegression( )
    self.lr.fit(self.x_train, self.y_train)

    # Testing Model
    # best possible score is 1
    self.lr_confidence = self.lr.score(self.x_test, self.y_test)
    print('lr confidence:', self.lr_confidence)

    # set x_forecast

    # self.x_forecast = np.array(self.df.drop(['Prognose'],1))[-97:]
    self.x_forecast = np.array(
        self.df.drop(['Prognose', 'Biomasse', 'Wasserkraft', 'WindOnshore', 'Kernenergie', 'Steinkohle'
                         , 'SonstigeKonventionelle', 'Erdgas', 'GesamteerzeugteEnergie', 'EEEnergie[MWh]',
                      'Anteil'], axis=1))
    # print(x_forecast)

    # Print linear Regression Model for the Prognose for n days
    self.lr_predection = self.lr.predict(self.x_forecast)
    print('lr predection', self.lr_predection)

    # Print support vector regressor model for the Prognose for n days
    self.svm_predection = self.svr_rbf.predict(self.x_forecast)
    print('svm pr', self.svm_predection)
    # self.t = pd.date_range(start=(self.df.index[len(self.df)-1]) , periods=24+1, freq='15min')
    # self.t1= pd.Series(self.t)
    # self.result2 = pd.Series(self.lr_predection)
    # self.result3=pd.DataFrame(columns = ['Date', 'Forecasting'])
    # self.result3['Date']=self.t1
    # self.result3['Forecasting']=self.result2
    # self.result3= self.result3[1:]
    # return self.result3

    # np.savetxt('./prognose/Prognose_Braunkohle.csv', self.lr_predection, delimiter=';')
    np.savetxt('Prognose_Braunkohle.csv', self.lr_predection, delimiter=';')
    print('ende', time.time( ) - timer)




    # plt.scatter(self.x_train, self.y_train, color='red')
    # plt.plot(self.x_train,self.svm_predection, color='blue')
    # plt.title('plot 1')
    # plt.show( )

    # Visualising the Test set results

    # plt.scatter(self.x_test, self.y_test, color='red')
    # plt.plot(self.x_train, self.svm_predection, color='blue')
    # plt.title('plot 2 ')
    # plt.show( )


if __name__ == '__main__':
    testetee = PrognoseBK( )
    timeind = SimConfig.timeindex
    print(timeind)
    testetee.run(timeind)

