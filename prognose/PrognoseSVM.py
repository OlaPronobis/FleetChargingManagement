#
#  creation and training of the svm regressor for the forecast that will be used in the
#  simulation, run this program one time before starting the simulation if the SimDur or
#  the time are new
#

import SimulationConfiguration as SimConfig
import prognose.PrognosePV_New as PrognosePV
import prognose.PrognoseSP_New as PrognoseSP
import prognose.PrognoseGL_New as PrognoseGL
import prognose.PrognoseCO2 as PrognoseCO2
import pandas as pd
import matplotlib.pyplot as plt


GL = PrognoseGL.PrognoseGL(fitting=True)
PV = PrognosePV.PrognosePV(fitting= True)
SP = PrognoseSP.PrognoseSP(fitting= True)
CO2 = PrognoseCO2.PrognoseCO2(fitting=True)

def fit_models(timeindex):
    if SimConfig.Forecast_building_load == 'yes':
        print('Forecasting building load...')
        GL.run(timeindex)
    if SimConfig.Forecast_PV == 'yes':
        print('Forecasting global solar radiation...')
        PV.run_forecast(timeindex)
    if SimConfig.Forecast_CO2 == 'yes':
        print('Forecasting CO2 emissions...')
        CO2.run(timeindex)
    if SimConfig.Forecast_current == 'yes':
        SP.run(timeindex)
        print('Forecasting electricity price...')



if __name__ == '__main__':
    if SimConfig.Level >= 2:
        start_date = SimConfig.time
        duration = SimConfig.SimDuration*96
        dates = pd.date_range(start=start_date, periods=duration, freq='15 min')
        for date in dates:
            date =str(date.strftime('%d.%m.%Y %H:%M'))
            print(date)
            fit_models(date)
        print('#########################')
        print('done')

    else:
        print("Level is 1")







