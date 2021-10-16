#https://medium.com/@randerson112358/predict-stock-prices-using-python-machine-learning-53aa024da20a

#Install the dependencies
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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

timer_all = time.time()

class PrognoseSP():

    def __init__(self, fitting = False, ):
        # read all the available data 'Zeitindex' 'Zeitwert' 'Monat' 'Strompreis' 'Vortagswert'
        self.df_full = pd.read_csv('./prognose/Eingangsdaten/Strompreis_final_test.csv', sep=';',
                                   parse_dates=True)
        # create a df of all the available data of the variable to be predicted
        #self.df_full_data = self.df_full[['Strompreis']]
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
        filename = './prognose/SVM/SP/' + t[0].replace('.', '_') + '_' + t[1].replace(':', 'h') + '_SP.pickle'

        # find the row of the timeindex
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # create a shorter df of the data (until the day+1 of the prediction because
        # the independent values (X) of the day to be forecasted are already known)
        # TODO zurzeit 2 years
        df = self.df_full[row - 96 * 365 * 2:row]
        # drop the irrelevant columns, new df columns:  'Zeitwert' 'Monat' 'Strompreis' 'Vortagswert'
        df = df.drop(['Zeitindex'], 1)

        # create scaler for X and y
        self.sc_X = StandardScaler( )
        self.sc_y = StandardScaler( )

        ### Create the independent data set (X) ###
        # df and array of the independent variables:  'Zeitwert' 'Monat' 'Vortagswert
        df_X = df.drop(['Strompreis'], 1)

        X = np.array(self.sc_X.fit_transform(df_X))
        # remove the last 97 rows for the training of the model
        X = X[:-self.forecast_out]

        ### Create the dependent data set (y)  #####
        # df and array of dependent variable 'Strompreis' [cent/kWh]
        df_y = df['Strompreis']

        y = np.array(df_y)
        y = self.sc_y.fit_transform((y.reshape(-1, 1)))
        # Get all of the y values except the last 97 rows
        y = y[:-self.forecast_out]

        # -----
        try:
            # try to load model, if there is already a svm for this timeindex use this one
            pickle_in = open(filename, 'rb')
            self.svr_rbf = pickle.load(pickle_in)
            print('SVR-SP for {} loaded'.format(timeindex))
        except FileNotFoundError:
            if self.fitting_model is False:
                warn("SVR-SP for {} doesn't exist, new SVM will be trained".format(timeindex))
            # weight the samples, the last month of the past year has a bigger weight
            # as well as the last 1 h and last day
            sample_weight = len(y) * [1]
            sample_weight[-15 * 96 - 365 * 96:-365 * 96 + 15 * 96] = [100] * 30 * 96
            sample_weight[- 96:] = [1000] * 96
            sample_weight[- 4:] = [5000] * 4

            # Create the regression model
            #self.svr_rbf = SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1,
            #              kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
            self.svr_rbf =SVR(C=0.1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.01,
                        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
            # train the model with the known data
            self.svr_rbf.fit(X, y, sample_weight=sample_weight)

            print('SP fitting time: ', time.time( ) - timer)

            # save fitted svr model
            with open(filename, 'wb') as f:
                pickle.dump(self.svr_rbf, f)
        #------

        # get the X values of the day to be forecasted
        x_forecast = np.array(df_X)[-self.forecast_out:]
        x_forecast = self.sc_X.transform(x_forecast)

        # forecast
        svm_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # set negetive predicted values to zero
        i=0
        for y in svm_prediction:
            if y < 0:
                svm_prediction[i] = 0
            i += 1

        # RMSE
        rmse = sqrt(mean_squared_error(df_y[-self.forecast_out:], svm_prediction))
        print('SP RMSE: ', rmse)

        # MAPE
        y_true = df_y[- self.forecast_out:]
        y_pred = svm_prediction
        try:
            mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except ZeroDivisionError:
            print('MAPE - zero division error')
            mape_val = np.mean(np.abs((y_true - y_pred) / (y_true+1e-3))) * 100
        print('SP MAPE: ', mape_val)


        # save prediction to csv
        np.savetxt('./prognose/Prognose_Strompreis.csv', svm_prediction, delimiter=';')



        # save prediction to new csv for evaluation - ausklammern wenn nicht benötigt
        # ToDo Erzeugen von csv zur MinMax Auswertung
        # filename = './prognose/Diagramm/SP_' + t[0].replace('.', '_') + '_' + t[1].replace(':', 'h') +'.csv'
        # np.savetxt(filename, svm_prediction, delimiter=';')


        # # plot
        # df_y[-self.forecast_out:].plot( )
        # plt.plot(df.index[- self.forecast_out:], svm_prediction, label='Prognostiziert')
        # plt.suptitle('RMSE: ' + str(rmse) + ' MAPE: ' + str(mape_val))
        # plt.legend( )
        # try:
        #     plt.savefig("C:\\Users\\Bubblebrote\\Desktop\\NProDia\\Strompreis\\0721_SimDur\\" +
        #                 str(timeindex[-5:]).replace(':', 'h'))
        # except FileNotFoundError:
        #     plt.show()

        return mape_val

    def get_new_values(self, timeindex):
        row = int(np.where(self.df_full['Zeitindex'] == timeindex)[0]) + self.forecast_out
        # get the X values of the day to be forecasted
        df_X = (self.df_full[row - self.forecast_out:row]).drop(['Zeitindex', 'Strompreis'], 1)
        x_forecast = np.array(self.sc_X.transform(df_X))

        # forecast
        svm_prediction = self.sc_y.inverse_transform(self.svr_rbf.predict(x_forecast))

        # save prediction to csv
        np.savetxt('./prognose/Prognose_Strompreis.csv', svm_prediction, delimiter=';')


if __name__ == '__main__':
    import datetime
    times = datetime.datetime(year=2019, month=4, day=15, hour=8, minute=0)
    # time = datetime.datetime(year=2019,month=1,day=24,hour=0,minute=15)
    # transform the time into form the same as the raw file of prognosis
    timeindex = str(times.strftime('%d.%m.%Y %H:%M'))

    prognose = PrognoseSP()
    prognose.run(timeindex)


#   def run(self, timeindex):
#         print('run time ', ti.time( ) - timer_all)
#         timer = ti.time( )
#         print(timeindex)
# #        timeindex = str(timeindex.strftime(
#  #           '%d.%m.%Y %H:%M'))  #
#         row = int(np.where(self.df_full['Daten'] == timeindex)[0])# + self.forecast_out
#         print('Row: ', row)
#
#         # create a shorter df of the data (until the day+1 of the prediction)
#         df = self.df_full_data[:row]
#         # Create another column (the target) shifted 96 units up
#         df['Prediction'] = df[['Strompreis']].shift(-self.forecast_out)
#
#         ### Create the independent data set (X) ###
#         # Convert the dataframe to a numpy array
#         X = np.array(df.drop(['Prediction'], 1))
#
#         # Remove the last '30' rows
#         X = X[:-self.forecast_out]
#
#         ### Create the dependent data set (y)  #####
#         # Convert the dataframe to a numpy array
#         y = np.array(df['Prediction'])
#         # Get all of the y values except the last 96 rows
#         y = y[:-self.forecast_out]
#
#         # Split the data into 80% training and 20% testing
#         x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#
#
#         # Create and train the Support Vector Machine (Regressor)
#         svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#         svr_rbf.fit(x_train, y_train)
#         print('time calc SP ', ti.time( ) - timer)
#
#         # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
#         # The best possible score is 1.0
#         #svm_confidence = svr_rbf.score(x_test, y_test)
#         #print("svm confidence SP: ", svm_confidence)
#
#         # ### Create and train the Linear Regression  Model (not used currently) ###
#         # lr = LinearRegression( )
#         # # Train the model
#         # lr.fit(x_train, y_train)
#         #
#         # # Testing Model: Score returns the coefficient of determination R^2 of the prediction.
#         # # The best possible score is 1.0
#         # lr_confidence = lr.score(x_test, y_test)
#         # print("lr confidence: ", lr_confidence)
#         # print('time calc lr', ti.time( ) - timer)
#         # ### ---
#
#         # Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
#         x_forecast = np.array(df.drop(['Prediction'], 1))[-self.forecast_out:]
#         #print(x_forecast)
#
#         # # Print linear regression model predictions for the next '30' days
#         # lr_prediction = lr.predict(x_forecast)
#         # print(lr_prediction)
#
#         # Print support vector regressor model predictions for the next 24h
#         svm_prediction = svr_rbf.predict(x_forecast)
#         #print(svm_prediction)
#
#
#         # RMSE
#         rmse = sqrt(mean_squared_error(self.df_full_data['Strompreis'][row - self.forecast_out:row], svm_prediction))
#         print('SP RMSE: ', rmse)
#
#         # RMSPE
#         y_true = self.df_full_data['Strompreis'][row - self.forecast_out:row]
#         y_pred = svm_prediction
#         try:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
#         except ZeroDivisionError:
#             rmspe_val = (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-100))))) * 100
#
#         print('SP RMSPE: ', rmspe_val)
#
#         # MAPE
#         try:
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         except ZeroDivisionError:
#             print('zero division error')
#             mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         print('SP MAPE: ', mape_val)
#
#         # plot
#         self.df_full_data[row - 97:row].plot( )
#         #plt.plot(self.df_full_data.index[row - 96:row], lr_prediction, color='green', label='lr')
#         plt.plot(self.df_full_data.index[row - 97:row], svm_prediction, color='red', label='svm')
#         #plt.legend( )
#         plt.show( )
#
#         # save prediction to csv
#         np.savetxt('./prognose/Prognose_Strompreis.csv', svm_prediction, delimiter=';')
#
#         return rmspe_val
#
