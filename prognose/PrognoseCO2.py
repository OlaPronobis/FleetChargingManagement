#Install the dependencies
import numpy as np
#from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas as pd
import  matplotlib.pyplot as plt
import time
import pickle
from warnings import warn

from sklearn.metrics import mean_squared_error
from math import sqrt

import SimulationConfiguration as SimConfig

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

class PrognoseCO2():
    def __init__(self, fitting = False):
        # read all the available data 'Zeitindex' 'Zeitwert' 'Monat' 'Jahreszeit' 'CO2' 'Vortagswert'
        self.df_full = pd.read_csv('./prognose/Eingangsdaten/CO2Emissionen_final_test.csv', sep=';',
                          parse_dates=True)
        # variable to store the last svm regressor
        self.svr_rbf = None
        # A variable for predicting 4*15=1 day out into the future
        self.forecast_out = SimConfig.Forecast_out + 1
        # variable if the svm fitting is needed
        self.fitting_model = fitting

    def run(self, timeindex):
        timer = time.time( )
        t = timeindex.split(' ')
        # filename of the pickle file of the svm
        filename = './prognose/SVM/CO2/' + t[0].replace('.', '_') + '_' + t[1].replace(':', 'h') + '_CO2.pickle'

        # find the row of the timeindex
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # create a shorter df of the data (until the day+1 of the prediction because
        # the independent values (X) of the day to be forecasted are already known)
        # TODO zurzeit 2 years
        df = self.df_full[row - 96 * 365 * 2:row]
        # drop the irrelevant columns, new df columns:  ''Zeitwert' 'Monat' 'Jahreszeit' 'CO2' 'Vortagswert'
        df = df.drop(['Zeitindex'], 1)

        # create scaler for X and y
        self.sc_X = StandardScaler( )
        self.sc_y = StandardScaler( )

        ### Create the independent data set (X) ###
        # df and array of the independent variables:  'Zeitwert' 'Monat' 'Jahreszeit' 'Vortagswert'
        df_X = df.drop(['CO2'], 1)

        X = np.array(self.sc_X.fit_transform(df_X))
        # remove the last 97 rows for the training of the model
        X = X[:-self.forecast_out]

        ### Create the dependent data set (y)  #####
        # df and array of dependent variable 'CO2' [g/kWh]
        df_y = df['CO2']

        y = np.array(df_y)
        y = self.sc_y.fit_transform((y.reshape(-1, 1)))
        # Get all of the y values except the last 97 rows
        y = y[:-self.forecast_out]

        # -----
        try:
            # try to load model, if there is already a svm for this timeindex use this one
            pickle_in = open(filename, 'rb')
            svr_rbf = pickle.load(pickle_in)
            print('SVR-CO2 for {} loaded'.format(timeindex))
        except FileNotFoundError:
            if self.fitting_model is False:
                warn("SVR-CO2 for {} doesn't exist, new SVM will be trained".format(timeindex))
            # weight the samples, the last month of the past year has a bigger weight
            # as well as the last 1 h and last day
            # alt:
            # sample_weight = len(y) * [1]
            # sample_weight[-15 * 96 - 365 * 96:-365 * 96 + 15 * 96] = [100] * 30 * 96
            # # sample_weight[-7 * 96:] = [500] * 7 * 96
            # sample_weight[- 96:] = [300] * 96
            # sample_weight[- 4:] = [2000] * 4
            # neu:
            sample_weight = len(y) * [1]
            sample_weight[-15 * 96 - 365 * 96:-365 * 96 + 15 * 96] = [100] * 30 * 96
            sample_weight[- 96:] = [1000] * 96
            sample_weight[- 4:] = [5000] * 4

            # Create the regression model
            # svr_rbf = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1,
            #              kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
            svr_rbf = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.1,
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

            # train the model with the known data
            svr_rbf.fit(X, y, sample_weight=sample_weight)

            print('CO2 fitting time: ', time.time( ) - timer)

            # save fitted svr model
            with open(filename, 'wb') as f:
                pickle.dump(svr_rbf, f)
        # ------

        # get the X values of the day to be forecasted
        x_forecast = np.array(df_X)[-self.forecast_out:]
        x_forecast = self.sc_X.transform(x_forecast)

        # forecast
        svm_prediction = self.sc_y.inverse_transform(svr_rbf.predict(x_forecast))

        # RMSE
        rmse = sqrt(mean_squared_error(df_y[-self.forecast_out:], svm_prediction))
        print('CO2 RMSE: ', rmse)

        # MAPE
        y_true = df_y[- self.forecast_out:]
        y_pred = svm_prediction
        try:
            mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except ZeroDivisionError:
            print('MAPE - zero division error')
            mape_val = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-3))) * 100
        print('CO2 MAPE: ', mape_val)

        # save prediction to csv
        np.savetxt('./prognose/Prognose_CO2.csv', svm_prediction, delimiter=';')


        # save prediction to new csv for evaluation - ausklammern wenn nicht benötigt
        # # ToDo Erzeugen von csv zur MinMax Auswertung
        # filename = './prognose/Diagramm/CO2_' + t[0].replace('.', '_') + '_' + t[1].replace(':', 'h') +'.csv'
        # np.savetxt(filename, svm_prediction, delimiter=';')


        # # plot
        # df_y[-self.forecast_out:].plot( )
        # plt.plot(df.index[- self.forecast_out:], svm_prediction, label='Prognostiziert')
        # plt.suptitle('RMSE: ' + str(rmse) + ' MAPE: ' + str(mape_val))
        # plt.legend( )
        # try:
        #     plt.savefig("C:\\Users\\Bubblebrote\\Desktop\\NProDia\\CO2Emission\\0723_SimDur\\" +
        #                 str(timeindex[-5:]).replace(':', 'h'))
        # except FileNotFoundError:
        #     plt.show( )

        return mape_val

    def get_new_values(self, timeindex):
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # get the X values of the day to be forecasted
        df_X = (self.df_full[row - self.forecast_out:row]).drop(['Zeitindex', 'CO2'], 1)
        x_forecast = np.array(self.sc_X.transform(df_X))

        # forecast
        svm_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # save prediction to csv
        np.savetxt('./prognose/Prognose_CO2.csv', svm_prediction, delimiter=';')




if __name__ == '__main__':
    import datetime
   # times = datetime.datetime(year=2019, month=4, day=8, hour=5, minute=0)
    times = datetime.datetime(year=2019, month=4, day=15, hour=20, minute=45)
    # time = datetime.datetime(year=2019,month=1,day=24,hour=0,minute=15)
    # transform the time into form the same as the raw file of prognosis
    timeindex = str(times.strftime('%d.%m.%Y %H:%M'))
    prognose = PrognoseCO2()
    prognose.run(timeindex)

#
# def forecast(timeindex, fitting = True):
#     t = timeindex.split(' ')
#     filename = './prognose/SVM/' + t[0].replace('.', '_') + t[1].replace(':', 'h') + 'CO2.pickle'
#     df_full = pd.read_csv('./prognose/Eingangsdaten/CO2Emissionen_final_test.csv', sep=';',
#                           parse_dates=True)
#     row = int(np.where(df_full['Zeitindex'] == timeindex)[0]) +96
#
#     df =df_full[row-96*913:row]
#
#     df = df.drop(['Zeitindex'],1)
#
#
#     # independent variable
#     df_X = df.drop(['CO2'],1)
#
#     # dependent variable
#     df_y = df['CO2']
#    # df_y[-30*96:].plot()
#    # plt.show()
#
#     sc_X = StandardScaler( )
#     sc_y = StandardScaler( )
#
#
#     X = np.array(sc_X.fit_transform(df_X))
#     #X = sc_X.fit_transform(X)
#
#     # Remove the last 97 rows
#     X = X[:-96]
#
#     y = np.array(df_y)
#     y = sc_y.fit_transform(y.reshape(-1, 1))
#     # Get all of the y values except the last 97 rows
#     y = y[:-96]
#     print('y', y)
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     #svr_rbf = SVR(kernel='rbf', C=10e3, gamma=0.001)
#     #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001)
#     #svr_rbf = SVR(kernel='rbf', C=1e4, gamma=0.1, epsilon=0.1)
#     if fitting is True:
#         svr_rbf = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1,
#                          kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#         sample_weight = len(y) * [1]
#         sample_weight[-15 * 96-365*96:-365*96+15*96] = [100] * 30 * 96 # monat in den vorjahren
#         #sample_weight[-7 * 96:] = [500] * 7 * 96
#         sample_weight[- 96:] = [1000]  * 96
#         sample_weight[- 4:] = [5000] * 4
#         print(len(sample_weight))
#        # sample_weight[-30 * 96 - 365*96*2:-365*96*2] = [1000] * 30 * 96
#         print(len(sample_weight))
#         svr_rbf.fit(X, y, sample_weight=sample_weight)
#         svm_confidence = svr_rbf.score(x_test, y_test)
#         print('conf ', svm_confidence)
#         # save fitted model
#         with open(filename, 'wb') as f:
#             pickle.dump(svr_rbf, f)
#
#     else:
#         # laden
#         pickle_in = open(filename, 'rb')
#         svr_rbf = pickle.load(pickle_in)
#
#     x_forecast = np.array(df_X[-96:])
#     print('xfore', x_forecast)
#
#     svm_prediction = sc_y.inverse_transform(svr_rbf.predict(sc_X.transform(x_forecast)))
#     svm_prediction_real = svm_prediction#[-96:]
#
#     print(svm_prediction_real)
#     # rmse
#     rmse = sqrt(mean_squared_error(df_y[-96:], svm_prediction_real))
#     # MAPE
#     y_true =df_y[-96:]
#     y_pred = svm_prediction_real
#     try:
#         mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     except ZeroDivisionError:
#         print('zero division error')
#         mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     print('RMSE: ', rmse)
#     print('GS MAPE: ', mape_val)
#
#     (df.drop(['Zeitwert', 'Monat', 'Jahreszeit', 'Vortag'], 1))[-96*3:].plot( )
#     plt.plot(df.index[- 96 :], svm_prediction)
#     plt.suptitle('RMSE: ' + str(rmse) + ' MAPE: ' + str(mape_val))
#     plt.legend( )
#     plt.show( )