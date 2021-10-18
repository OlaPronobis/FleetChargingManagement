import datetime
import time as t
import pandas as pd
import numpy as np
#STANDORT-----------------------------------------------------
#set up of the area of PV
Area_PV = 700 #m^2
#set up of the efficiency of PV
Efficiency_PV = 10 #%
#set up of the illumination angle
Illumination_Angle = 32 #°
#set up of grid value
Grid =138.92 #254.84 #138.92 #80.96 #kW #87.96 #68.64 #58.98 #38.28

#------------------------------------------------------------------

#FAHRZEUGE-----------------------------------------------------
#set up of the number of EVs of ESD
Number_ESD = 12
# set up of the battery energy of EV
E_EV = 94  # kWh
# set up of the maximum charging power
P_max = 11  # kW
#set up of the beginning SoC of each EV

SOC_Start = [50]*Number_ESD

#set up of the number of EVs of KED
Number_KED = 24
# set up of the battery energy of KED
E_KED = 35.8
# set up of the charging power of each KED
CP_KED = 2.76  # kW
#set up the SOC of each KED EV
# randomint(lowerlimit, upperlimit, number_of_cars)
# SOC_KED = np.random.randint(60, 70, Number_KED).tolist()
SOC_KED = [100] * Number_KED
#set up a soc buffer
SOC_buffer = 0.5

#Standing time of each ESD at charging station - Für Level 2 und 1
Standing_time = 2.6 #hr #2.60184 #2.6
#Driving time of each ESD
Driving_time = 1.4 #hr #1.430512 #1.4
#consumed SOC during the driving time
SOC_Consumed = 15.6711 #%
#-----------------------------------------------------------------------------------------------------

#SIMULATIONSEINSTELLUNGEN-------------------------------------------------------
# set up of the simulation begin
time = datetime.datetime(year=2019, month=4, day=15, hour=0, minute=0)
# transform the time into form the same as the raw file of prognosis
timeindex = str(time.day) + '.' + str(time.month) + '.' + str(time.year) + ' ' + str(time.hour) + ':' + str(time.minute)
# transform the timeindex into timestamp
timeArray = t.strptime(timeindex, "%d.%m.%Y %H:%M")
start_timestamp = t.mktime(timeArray)

#set up the simulation duration
SimDuration = 5 #days


#optimization target('Cost'/'Emission'), Coefficient for ZF (NOTE: C_Emission + C_Cost == 1)
OTM_Target = 'Emission' # todo coeff einbidn und otm target abfrage mainsim
C_Emission = 0
C_Cost = 1

# static charge algo, if no optimazation is required -> 1, Level should be set to 1
No_Opt = 0
# algorithm ('EQUAL'/'FCFS')
Static_Algo = 'FCFS'

#define complexity level (3/2/1)
Level = 1

nu = '209xxx'
da = 200925

#if you choose level 1, define the forecasts needed ('yes'/'no')
# if the level is 1, at least one of the following forecast should be set as 'no'
Forecast_CO2 = 'no'
Forecast_current = 'no'
Forecast_PV = 'no'
Forecast_building_load = 'no'

# Forecast_CO2 = 'yes'
# Forecast_current = 'yes'
# Forecast_PV = 'yes'
# Forecast_building_load = 'yes'


Forecast_out = 96 #wie viele 15 Min-Werte
#---------------------------------------------------------------------------------------------------

