#https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a

#Install the dependencies
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import  matplotlib.pyplot as plt
import datetime
import time
import math
import pickle
from warnings import warn

from sklearn.metrics import mean_squared_error

import SimulationConfiguration as SimConfig


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

class PrognosePV():

    def __init__(self, fitting = False):
        # data that will be used if the forecast function is deactivated
        self.df_no_forecast = pd.read_excel('./prognose/Eingangsdaten/200610_Prognose Globalstrahlung_fertig.xlsx')

        # read all the available data 'Zeitindex' 'Zeitwert' 'Monat' 'Jahreszeit' 'Globalstrahlung' [W/m^2]
        self.df_full = pd.read_csv('./prognose/Eingangsdaten/Globalstrahlung_final.csv', sep=';',
                                   parse_dates=True)
        # variable if the svm fitting is needed
        self.fitting_model = fitting
        # variable to store the last svm regressor
        self.svr_rbf = None
        # A variable for predicting 4*15=1 day out into the future
        self.forecast_out = SimConfig.Forecast_out + 1
        # PV config
        self.area= SimConfig.Area_PV
        self.efficiency = SimConfig.Efficiency_PV
        self.angle = SimConfig.Illumination_Angle
        self.cos = math.cos(math.pi * self.angle / 180)


    def run(self, timeindex, new_svr = True):
        if SimConfig.Forecast_PV == 'no' or SimConfig.Level == 1:

            timeArray = time.strptime(timeindex, "%d.%m.%Y %H:%M")
            timestamp = time.mktime(timeArray)
            time_local = time.localtime(timestamp)
            dt = datetime.datetime(year = 2010, month=time_local.tm_mon,day = time_local.tm_mday)
            t = datetime.time(hour=time_local.tm_hour,minute=time_local.tm_min)
            df1 = self.df_no_forecast[self.df_no_forecast.iloc[:,0] == dt]
            df1 = df1[df1.iloc[:,1] == t]
            a = df1.index[0]
            PV= []
            for i in range(a,a+self.forecast_out):
                GS = self.df_no_forecast.iloc[i,8]
                PV_power = GS*self.cos*0.01*self.efficiency*self.area*0.001
                PV.append(PV_power)
            df_save = pd.DataFrame(data=PV)
            df_save.to_csv('./prognose/Prognose_PVPower.csv',header = False, index = False)
        else:
            if new_svr is False and self.svr_rbf is not None:
                self.get_new_values(timeindex)
            else:
                self.run_forecast(timeindex)

    # new forecast
    def run_forecast(self, timeindex):
        timer = time.time( )
        t = timeindex.split(' ')
        # filename of the pickle file of the svm
        filename = './prognose/SVM/GS/' + t[0].replace('.', '_') +'_'+ t[1].replace(':', 'h') + '_GS.pickle'

        # find the row of the timeindex
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # create a shorter df of the data (until the day+1 of the prediction because
        # the independent values (X) of the day to be forecasted are already known)
        # TODO zurzeit 2 years
        df = self.df_full[row-96*365*2:row]
        # drop the irrelevant columns, new df columns:  'Zeitwert' 'Monat' 'Jahreszeit' 'Globalstrahlung'
        df = df.drop(['Zeitindex'], 1)

        # create scaler for X and y
        self.sc_X = StandardScaler( )
        self.sc_y = StandardScaler( )

        ### Create the independent data set (X) ###
        # df and array of the independent variables:  'Zeitwert' 'Monat' 'Jahreszeit'
        df_X = df.drop(['Globalstrahlung'], 1)

        X = np.array(self.sc_X.fit_transform(df_X))
        # remove the last 97 rows for the training of the model
        X = X[:-self.forecast_out]

        ### Create the dependent data set (y)  #####
        # df and array of dependent variable 'Globalstrahlung' [W/m^2]
        df_y = df['Globalstrahlung']

        y = np.array(df_y)
        y = self.sc_y.fit_transform((y.reshape(-1, 1)))
        # Get all of the y values except the last 97 rows
        y = y[:-self.forecast_out]

        # -----
        try:
            # try to load model, if there is already a svm for this timeindex use this one
            pickle_in = open(filename, 'rb')
            self.svr_rbf = pickle.load(pickle_in)
            print('SVR-GS for {} loaded'.format(timeindex))
        except FileNotFoundError:
            if self.fitting_model is False:
                warn("SVR-GS for {} doesn't exist, new SVM will be trained".format(timeindex))
            # weight the samples, the last month of the past year has a bigger weight
            # as well as the last 1-1,5 h
            sample_weight = len(y) * [1]
            sample_weight[-15 * 96 - 365 * 96:-365 * 96 + 15 * 96] = [100] * 30 * 96
            #sample_weight[-6:-4] = [500] * 2
            sample_weight[-4:] = [500] * 4

            # Create the regression model
            self.svr_rbf = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1,
                          kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
            # train the model with the known data
            self.svr_rbf.fit(X, y, sample_weight=sample_weight)

            print('PV fitting time: ', time.time( ) - timer)

            # save fitted svr model
            with open(filename, 'wb') as f:
                pickle.dump(self.svr_rbf, f)
        #------

        # get the X values of the day to be forecasted
        x_forecast = np.array(df_X)[-self.forecast_out:]
        x_forecast = self.sc_X.transform(x_forecast)

        # forecast
        svm_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # set values between [20h30:03h45] to zero
        time_line = np.where((df_X[-self.forecast_out:])['Zeitwert'] < 0.17)[0]
        time_line = np.append(time_line, np.where((df_X[-self.forecast_out:])['Zeitwert'] > 0.84)[0])
        for i in time_line:
            svm_prediction[i] = 0

        # set negetive predicted values to zero
        i=0
        for y in svm_prediction:
            if y < 0:
                svm_prediction[i] = 0
            i += 1

        #print(svm_prediction)

        # RMSE
        rmse = math.sqrt(mean_squared_error(df_y[-self.forecast_out:], svm_prediction))
        print('PV RMSE: ', rmse)

        # MAPE
        # row range where the value is over 1 w/m^2
        x_start = np.where(df_y[- self.forecast_out:] > 1)[0][0]
        x_end = np.where(df_y[- self.forecast_out:] > 1)[0][-1]
        y_true = df_y[- self.forecast_out:]
        y_true = y_true[x_start:x_end]
        y_pred = svm_prediction[x_start:x_end]
        try:
            mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except ZeroDivisionError:
            print('MAPE - zero division error')
            mape_val = np.mean(np.abs((y_true - y_pred) / (y_true+1e-3))) * 100
        print('GS MAPE: ', mape_val)

        # calculate PV power [kW] '
        PV_power = svm_prediction * self.cos * 0.01 * self.efficiency * self.area * 0.001
        #print(PV_power)

        # save prediction to csv
        np.savetxt('./prognose/Prognose_PVPower.csv', PV_power, delimiter=';')

        # # plot
        # df_y[-self.forecast_out:].plot( )
        # plt.plot(df.index[- self.forecast_out:], svm_prediction, label='Prognostiziert')
        # plt.suptitle('RMSE: ' + str(rmse) + ' MAPE: ' + str(mape_val))
        # plt.legend( )
        # # plt.show()
        # #
        # plt.savefig("C:\\Users\\Bubblebrote\\Desktop\\NProDia\\Globalstrahlung\\0721_SimDur\\" +
        #             str(timeindex[-5:]).replace(':','h')+'_2')

        return mape_val

    def get_new_values(self, timeindex):
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # get the X values of the day to be forecasted
        df_X = (self.df_full[row-self.forecast_out:row]).drop(['Zeitindex', 'Globalstrahlung'], 1)
        x_forecast = np.array(self.sc_X.transform(df_X))

        # forecast
        svm_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # set values between [20h30:03h45] to zero
        time_line = np.where((df_X[-self.forecast_out:])['Zeitwert'] < 0.17)[0]
        time_line = np.append(time_line, np.where((df_X[-self.forecast_out:])['Zeitwert'] > 0.84)[0])
        for i in time_line:
            svm_prediction[i] = 0

        # set negetive predicted values to zero
        i = 0
        for y in svm_prediction:
            if y < 0:
                svm_prediction[i] = 0
            i += 1

        # calculate PV power [kW] '
        PV_power = svm_prediction * self.cos * 0.01 * self.efficiency * self.area * 0.001

        # save prediction to csv
        np.savetxt('./prognose/Prognose_PVPower.csv', PV_power, delimiter=';')






if __name__ == '__main__':
    times = datetime.datetime(year=2019, month=4, day=16, hour=6, minute=0)
    # time = datetime.datetime(year=2019,month=1,day=24,hour=0,minute=15)
    # transform the time into form the same as the raw file of prognosis
    timeindex = str(times.strftime('%d.%m.%Y %H:%M'))

    prognose = PrognosePV()
    prognose.run(timeindex, new_svr=True)
    times = datetime.datetime(year=2019, month=4, day=16, hour=9, minute=15)
    # time = datetime.datetime(year=2019,month=1,day=24,hour=0,minute=15)
    # transform the time into form the same as the raw file of prognosis
    timeindex = str(times.strftime('%d.%m.%Y %H:%M'))
    prognose.run(timeindex)


#
# def run_forecast_old(self, timeindex):
#         timer = time.time( )
#         row = int(np.where(self.df_full['Daten'] == timeindex)[0]) + self.forecast_out
#         # create a shorter df of the data (until the day+1 of the prediction)
#         df = self.df_full_data[:row]
#         # Create another column (the target) shifted 97 units up
#         df['Prediction'] = df[['Globalstrahlung']].shift(-self.forecast_out)
#
#         ### Create the independent data set (X) ###
#         # Convert the dataframe to a numpy array
#         X = np.array(df.drop(['Prediction'], 1))
#
#         # Remove the last 97 rows
#         X = X[:-self.forecast_out]
#
#         ### Create the dependent data set (y)  #####
#         # Convert the dataframe to a numpy array
#         y = np.array(df['Prediction'])
#         # Get all of the y values except the last 97 rows
#         y = y[:-self.forecast_out]
#
#         # Split the data into 80% training and 20% testing
#         x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#         # Create and train the Support Vector Machine (Regressor) NOTE: not used
#         svr_rbf = SVR(kernel='rbf', C=1e2, gamma='scale')
#         svr_rbf.fit(x_train, y_train)
#         print('time calc GS:', time.time( ) - timer)
#
#         # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
#         # The best possible score is 1.0
#         svm_confidence = svr_rbf.score(x_test, y_test)
#         print("svm confidence: ", svm_confidence)
#
#         # ### Create and train the Linear Regression  Model ###
#         # lr = LinearRegression( )
#         # # Train the model
#         # lr.fit(x_train, y_train)
#         #
#         # # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
#         # # The best possible score is 1.0
#         # lr_confidence = lr.score(x_test, y_test)
#         # print("lr confidence: ", lr_confidence)
#         # print('time calc lr', time.time( ) - timer)
#         # ### ---
#
#         # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
#         x_forecast = np.array(df.drop(['Prediction'], 1))[-self.forecast_out:]
#         # print(x_forecast)
#
#         # Print linear regression model predictions for the next 24h
#         # lr_prediction = lr.predict(x_forecast)
#         # print(lr_prediction)
#
#         # Print support vector regressor model predictions for the next 24h
#         svm_prediction = svr_rbf.predict(x_forecast)
#         # print(svm_prediction)
#
#         # RMSE
#         rmse = math.sqrt(
#             mean_squared_error(self.df_full_data['Globalstrahlung'][row - self.forecast_out:row], svm_prediction))
#         print('GS RMSE: ', rmse)
#
#         # RMSPE
#         # row range where the value is over 1 w/m^2
#         x_start = np.where(self.df_full['Globalstrahlung'][row - 97:row] > 1)[0][0]
#         x_end = np.where(self.df_full['Globalstrahlung'][row - 97:row] > 1)[0][-1]
#         y_true = self.df_full_data['Globalstrahlung'][row - 97:row]
#         y_true = y_true[x_start:x_end]
#         y_pred = svm_prediction[x_start:x_end]
#         try:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
#         except ZeroDivisionError:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-100))))) * 100
#         print('GS RMSPE: ', rmspe_val)
#
#         # MAPE
#         try:
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         except ZeroDivisionError:
#             print('zero division error')
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         print('GS MAPE: ', mape_val)
#
#         # plot
#         self.df_full_data[row - self.forecast_out:row].plot( )
#         # plt.plot(self.df_full_data.index[row - 96:row], lr_prediction, color='green', label='lr')
#         # plt.plot(self.df_full_data.index[row - 97:row], svm_prediction, color='red', label='svm')
#         plt.scatter(self.df_full_data.index[row - 97:row], svm_prediction)
#         # plt.legend( )
#         plt.show( )
#
#         # calculate PV power [kW] '
#         PV_power = svm_prediction * self.cos * 0.01 * self.efficiency * self.area * 0.001
#
#         # save prediction to csv
#         np.savetxt('./prognose/Prognose_PVPower.csv', PV_power, delimiter=';')
#
#         return rmspe_val