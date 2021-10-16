#https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a

import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import time
import pickle
from warnings import warn

from sklearn.metrics import mean_squared_error
from math import sqrt

import SimulationConfiguration as SimConfig


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

class PrognoseGL():

    def __init__(self, fitting=False):
        # data that will be used if the forecast is deactivated
        self.df_no_forecast = pd.read_excel('./prognose/Eingangsdaten/200606_SLP-G3_Niedersachsenbearbeitet.xlsm')

        # read all the available data
        self.df_full = pd.read_csv('./prognose/Eingangsdaten/Gebaeudelast_1819_final.csv', sep=';',
                                   parse_dates=True)
        # variable if the svm fitting is needed
        self.fitting_model = fitting
        # variable to store the last svm regressor
        self.svr_rbf = None
        # A variable for predicting 4*15=1 day out into the future
        self.forecast_out = SimConfig.Forecast_out + 1

    def run(self, timeindex, new_svr = True):
        if SimConfig.Forecast_building_load == 'no' or SimConfig.Level ==1:
            timeArray = time.strptime(timeindex, "%d.%m.%Y %H:%M")
            timestamp = time.mktime(timeArray)
            time_local = time.localtime(timestamp)
            dt = datetime.datetime(year=2018, month=time_local.tm_mon, day=time_local.tm_mday)
            t = datetime.time(hour=time_local.tm_hour, minute=time_local.tm_min)
            df1 = self.df_no_forecast[self.df_no_forecast.iloc[:, 3] == dt]
            df1 = df1[df1.iloc[:, 4] == t]
            a = df1.index[0]
            GL = []
            for i in range(a, a + SimConfig.Forecast_out+1):
                PW = self.df_no_forecast.iloc[i, 7]
                GL.append(PW)
            df_save = pd.DataFrame(data=GL)
            df_save.to_csv('./prognose/Prognose_Gebaeudelast.csv', header=False, index=False)

        else:
            if new_svr is False and self.svr_rbf is not None:
                self.get_new_values(timeindex)
            else:
                self.run_forecast(timeindex)


    def run_forecast(self, timeindex):
        timer = time.time( )
        t = timeindex.split(' ')
        # filename of the pickle file of the svm
        filename = './prognose/SVM/GL/' + t[0].replace('.', '_') + '_' + t[1].replace(':', 'h') + '_GL.pickle'

        # find the row of the timeindex
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # create a shorter df of the data (until the day+1 of the prediction because
        # the independent values (X) of the day to be forecasted are already known)
        df = self.df_full[:row]
        # drop the irrelevant columns, new df columns:  'Zeitwert' 'Monat' 'Jahreszeit' 'Globalstrahlung'
        df = df.drop(['Zeitindex'], 1)

        # create scaler for X and y
        self.sc_X = StandardScaler( )
        self.sc_y = StandardScaler( )

        ### Create the independent data set (X) ###
        # df and array of the independent variables: 'Zeitwert' 'Wochentag'
        df_X = df.drop(['kW'], 1)
        #print(df_X)

        X = np.array(self.sc_X.fit_transform(df_X))
        # remove the last 97 rows for the training of the model
        X = X[:-self.forecast_out]

        ### Create the dependent data set (y)  #####
        # df and array of dependent variable 'kW'
        df_y = df['kW']

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

            # weight the samples, the last month has a bigger weight and the last same weekday has the most weight
            sample_weight = len(y) * [1]
            sample_weight[-30 * 96:] = [10] * 30 * 96
            sample_weight[-7 * 96:-6 * 96] = [2000] * 96
            sample_weight[-4:] = [3000] * 4

            # Create the regression model
          #  self.svr_rbf = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.01,
          #              kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
            self.svr_rbf =SVR()
            # train the model with the known data
            self.svr_rbf.fit(X, y, sample_weight=sample_weight)

            print('GL fitting time: ', time.time( ) - timer)

            # save fitted svr model
            with open(filename, 'wb') as f:
                pickle.dump(self.svr_rbf, f)
        # ------

        # get the X values of the day to be forecasted
        x_forecast = np.array(df_X)[-self.forecast_out:]
        x_forecast = self.sc_X.transform(x_forecast)

        # forecast
        lr_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # RMSE
        rmse = sqrt(mean_squared_error(df_y[-self.forecast_out:], lr_prediction))
        print('GL RMSE: ', rmse)

        # RMSPE
        y_true = np.array(df_y[-self.forecast_out:])
        y_pred = lr_prediction
        try:
            rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
        except ZeroDivisionError:
            print('zero division error')
            rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-100))))) * 100

        print('GL RMSPE: ', rmspe_val)

        # MAPE
        try:
            mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except ZeroDivisionError:
            print('zero division error')
            mape_val = np.mean(np.abs((y_true - y_pred) / y_true + 1e-100)) * 100
        print('GL MAPE: ', mape_val)

        # save prediction to csv
        np.savetxt('./prognose/Prognose_Gebaeudelast.csv', lr_prediction, delimiter=';')

        # # save prediction to new csv for evaluation - ausklammern wenn nicht benötigt
        # filename = './prognose/Diagramm/GL_' + t[0].replace('.', '_') + '_' + t[1].replace(':', 'h') +'.csv'
        # np.savetxt(filename, lr_prediction, delimiter=';')

        # plot
        # df_y[-self.forecast_out:].plot( )
        # plt.plot(df.index[- self.forecast_out:], lr_prediction, label='Prognostiziert')
        # plt.suptitle('RMSE: ' + str(rmse) + ' MAPE: ' + str(mape_val))
        # plt.legend( )
        # try:
        #  plt.savefig("C:\\Users\\Bubblebrote\\Desktop\\NProDia\\Gebaudelast\\0721_SimDur\\" +
        #                  str(timeindex[-5:]).replace(':', 'h'))
        # except FileNotFoundError:
        #      plt.show()
        #
        #
        # return mape_val


    def get_new_values(self,timeindex):
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # get the X values of the day to be forecasted
        df_X = (self.df_full[row - self.forecast_out:row]).drop(['Zeitindex', 'kW'], 1)
        x_forecast = np.array(self.sc_X.transform(df_X))

        # forecast
        svm_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # save prediction to csv
        np.savetxt('./prognose/Prognose_Gebaeudelast.csv', svm_prediction, delimiter=';')


if __name__ == '__main__':
    import datetime
    times = datetime.datetime(year=2019, month=4, day=1, hour=0, minute=0)
    # time = datetime.datetime(year=2019,month=1,day=24,hour=0,minute=15)
    # transform the time into form the same as the raw file of prognosis
    timeindex = str(times.strftime('%d.%m.%Y %H:%M'))

    prognose = PrognoseGL()
    prognose.run_forecast(timeindex)


# def old_(self,timeindex):
#         t0 = timeindex.split(' ')
#         # find the date of the timeindex t0 and then the time
#         date_row = np.where(self.df_full['Datum'] == t0[0])[0]
#         row = None
#         for x in date_row:
#             if self.df_full['Zeit'][x] == (t0[1] + str(':00')):
#                 row = x + self.forecast_out
#         # create a shorter df of the data (until the day+1 of the prediction because
#         # the independent values (X) of the day to be forecasted are already known)
#         df = self.df_full[:row]
#         # drop the irrelevant columns, new df columns: 'Profilwert' 'Zeitwert' 'Wochentag' 'kW'
#         df = df.drop(['Datum', 'Zeit'],1)
#
#         # create scaler for X and y
#         sc_X = StandardScaler()
#         sc_y = StandardScaler()
#
#         ### Create the independent data set (X) ###
#         # df and array of the independent variables: 'Profilwert' 'Zeitwert' 'Wochentag'
#         df_X =df.drop(['kW'],1)
#
#         X = np.array(sc_X.fit_transform(df_X))
#         # remove the last 97 rows for the training of the model
#         X = X[:-self.forecast_out]
#
#         ### Create the dependent data set (y)  #####
#         # df and array of dependent variable 'kW'
#         df_y = df['kW']
#
#         y = np.array(df_y)
#         y = sc_y.fit_transform((y.reshape(-1,1)))
#         # Get all of the y values except the last 97 rows
#         y = y[:-self.forecast_out]
#
#         # weight the samples, the last month has a bigger weight and the last same weekday has the most weight
#         sample_weight = len(y) * [1]
#         sample_weight[-30*96:] = [10]*30*96
#         sample_weight[-7*96:-6*96] = [1000]*96
#
#         # Create the regression model
#        # lr = LinearRegression()
#         lr = SVR(C=10e3, gamma= 0.1)
#         # train the model with the known data
#         lr.fit(X,y, sample_weight=sample_weight)
#
#         print('GL fitting time: ', time.time() - timer)
#
#         # get the X values of the day to be forecasted
#         x_forecast = np.array(df_X)[-self.forecast_out:]
#         x_forecast = sc_X.transform(x_forecast)
#
#         # forecast
#         lr_prediction = sc_y.inverse_transform(lr.predict(x_forecast))
#
#         # RMSE
#         rmse = sqrt(mean_squared_error(df_y[-self.forecast_out:], lr_prediction))
#         print('GL RMSE: ', rmse)
#
#         # RMSPE
#         y_true = np.array(df_y[-self.forecast_out:])
#         y_pred = lr_prediction
#         try:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
#         except ZeroDivisionError:
#             print('zero division error')
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-100))))) * 100
#
#         print('GL RMSPE: ', rmspe_val)
#
#         # MAPE
#         try:
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         except ZeroDivisionError:
#             print('zero division error')
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true+1e-100)) * 100
#         print('GL MAPE: ', mape_val)
#
#         # save prediction to csv
#         np.savetxt('./prognose/Prognose_Gebaeudelast.csv', lr_prediction, delimiter=';')
#
#         # plot
#         df_y[-self.forecast_out:].plot( )
#         plt.plot(df.index[- self.forecast_out:], lr_prediction, label='Prognostiziert')
#         plt.suptitle('RMSE: ' + str(rmse) + ' MAPE: ' + str(mape_val))
#         plt.legend( )
#
#         # plt.close()
#         plt.show()
#
#         return mape_val
#
#
# def alt(self):
#         row = int(np.where(self.df_full['Daten'] == timeindex)[0]) + self.forecast_out
#         # create a shorter df of the data (until the day+1 of the prediction)
#         df = self.df_full_data[:row]
#         # Create another column (the target) shifted 97 units up
#         df['Prediction'] = df[['kWh']].shift(-self.forecast_out)
#
#
#         # Convert the dataframe to a numpy array
#         X = np.array(df.drop(['Prediction'], 1))
#
#         # Remove the last 97 rows
#         X = X[:-self.forecast_out]
#
#
#         # Convert the dataframe to a numpy array
#         y = np.array(df['Prediction'])
#         # Get all of the y values except the last 97 rows
#         y = y[:-self.forecast_out]
#
#         # Split the data into 80% training and 20% testing
#         x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#         # # Create and train the Support Vector Machine (Regressor) NOTE: not used
#         # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#         # svr_rbf.fit(x_train, y_train)
#         # print('time calc svm ', ti.time( ) - timer)
#         #
#         # # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
#         # # The best possible score is 1.0
#         # svm_confidence = svr_rbf.score(x_test, y_test)
#         # print("svm confidence: ", svm_confidence)
#
#         ### Create and train the Linear Regression  Model ###
#         lr = LinearRegression( )
#         # Train the model
#         lr.fit(x_train, y_train)
#
#         # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
#         # The best possible score is 1.0
#         lr_confidence = lr.score(x_test, y_test)
#         print("lr confidence GL: ", lr_confidence)
#         print('time calc GL', time.time( ) - timer)
#         ### ---
#
#         # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
#         x_forecast = np.array(df.drop(['Prediction'], 1))[-self.forecast_out:]
#         #print(x_forecast)
#
#         # Print linear regression model predictions for the next 24h
#         lr_prediction = lr.predict(x_forecast)
#         # print(lr_prediction)
#
#         # Print support vector regressor model predictions for the next 24h
#         # svm_prediction = svr_rbf.predict(x_forecast)
#         # print(svm_prediction)
#
#         # RMSE
#         rmse = sqrt(mean_squared_error(self.df_full_data['kWh'][row - self.forecast_out:row], lr_prediction))
#         #rmse = sqrt(mean_squared_error(self.df_full_data['kWh'][row:row+self.forecast_out], lr_prediction))
#         print('GL RMSE: ', rmse)
#
#
#         # RMSPE
#         y_true = self.df_full_data['kWh'][row - self.forecast_out:row]
#         #y_true = self.df_full_data['kWh'][row:row+self.forecast_out]
#         y_pred = lr_prediction
#         try:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
#         except ZeroDivisionError:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-100))))) * 100
#
#         print('GL RMSPE: ', rmspe_val)
#
#         # MAPE
#         try:
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         except ZeroDivisionError:
#             print('zero division error')
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         print('GL MAPE: ', mape_val)
#
#         # plot
#        # print(lr_prediction)
#        # print(y_true)
#         self.df_full_data[row- self.forecast_out:row].plot( )
#         plt.plot(self.df_full_data.index[row-self.forecast_out:row], lr_prediction, color='green', label='lr')
#         #self.df_full_data[row:row + self.forecast_out].plot( )
#         #plt.plot(self.df_full_data.index[row:row+self.forecast_out], lr_prediction, color='green', label='lr')
#         #plt.plot(self.df_full_data.index[row - 97:row], svm_prediction, color='red', label='svm')
#         #plt.legend( )
#         plt.show( )
#
#         # save prediction to csv
#         np.savetxt('./prognose/Prognose_Gebaeudelast.csv', lr_prediction, delimiter=';')
#
#
#         #test
#         #xforecast = np.array(self.df_full_data[row+97*2:row+97*3] )#[-self.forecast_out:]np.array(self.df_full_data.value[row:row+97])
#         #lr_prediction = lr.predict(xforecast)
#
#         #self.df_full_data[row+97*2:row+97*3].plot( )
#         #plt.plot(self.df_full_data.index[row+97*2:row+97*3], lr_prediction, color='green', label='lr')
#         #plt.show()
#
#
#
#         return rmspe_val