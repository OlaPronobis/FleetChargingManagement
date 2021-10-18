import pandas as pd
import numpy as np
import time
import os
import datetime
import math
import random
import csv

import main_prognose_1p as Prognose1p
import SimulationConfiguration as SimConfig
import prognose2p.main_prognose_2p as Prognose2p
import GestaltungFahrplan as GestaltungFahrplan
import GestaltungFahrplanAfterCharging as GestaltungFahrplanAfterCharging
import Optimization as Optimization
import matplotlib.pyplot as plt
from matplotlib.figure import SubplotParams


def studentisierung(dataframe_cost, dataframe_co2):
    results = []

    #Normierung
    for df in [dataframe_cost, dataframe_co2]:
        # 1a. Min und Max
        data_min = df.min()
        data_max = df.max()

        # 1b. Differenz
        df_m = df - data_min
        # 1c. Normiert
        df_q = df_m / ((1 / 10000) * (data_max - data_min))
        results.append(df_q)


    return results[0], results[1]


# KED
# set up KED EVs
SOC_KED = SimConfig.SOC_KED
df_KED = pd.read_csv('./TestScenario/KED/KED_final.csv')


# fkt to get the current number of charging ked
def get_current_ked_number(timeindex):
    row = int(np.where(df_KED['Timestamp'] == timeindex)[0])
    ked_number = 0
    # iterate over all KED
    for x in range(1, SimConfig.Number_KED + 1):
        ked_state = df_KED.loc[row, 'Charge_EV_%s' % x]
        # counting number of KED with charge status is 1 and whose SOC is below 100
        if ked_state and SOC_KED[x - 1] < 100:
            ked_number += 1
            # if charging then add the next
            SOC_KED[x - 1] = SOC_KED[x - 1] + (((1 / 4) * CP_KED) / E_KED) * 100
            if SOC_KED[x - 1] > 100:
                SOC_KED[x - 1] = 100
        elif df_KED.loc[row - 1, 'Charge_EV_%s' % x] == 1 and ked_state == 0:
            SOC_KED[x - 1] = SOC_KED[x - 1] - df_KED.loc[row, 'SOC_Usage_EV_%s' % x]
    return ked_number


# data initialization
forecast_out = SimConfig.Forecast_out
CS = []
dfempty = []
ChargingPower = []
TimeCharging = []
SOC_Start = SimConfig.SOC_Start
SOC_List = []
y_CP_stacked = []
Price = []
PriceSum = []
Emission = []
EmissionSum = []
xlabel = []
EV_charging_ID = []
EV_charging_ID_1 = []
Index_begin = {}
SOC_to_charge_sum = 0
Load_limit = []
load_limit_PV = []
load_limit_GL = []
load_limit_KED = []
y_LL_stacked = []
SOC_cache = []
SOC_before = []
xlabel_ex = []
index_SOC_1 = {}
SOC_1 = {}
SOC_average_1 = {}
list_charging_evs_with_prio = []
# list to save if ev is at the charging station
Station = []

# data import
Number_EVs = SimConfig.Number_ESD
E_EV = SimConfig.E_EV
OTM_Target = SimConfig.OTM_Target
P_max = SimConfig.P_max
E_EV = SimConfig.E_EV
Number_KED = SimConfig.Number_KED
CP_KED = SimConfig.CP_KED
E_KED = SimConfig.E_KED

# data initialization of the lists
power_1st = [0] * Number_EVs
SOC_Now = SOC_Start  # [0]*Number_EVs
SOC_cache = [0] * Number_EVs
SOC_before = [0] * Number_EVs
chargingpower = [0] * Number_EVs
SOC_rest = [0] * Number_EVs

# Factors of energie source
FaktorBK = 407
FaktorSK = 336
FaktorEG = 201
FaktorSon = 288

# initialize the 2D-Lists
for i in range(Number_EVs):
    ChargingPower.append([0])
    SOC_List.append([0])
    Price.append([0])
    Emission.append([0])
    y_CP_stacked.append([0])
    PriceSum.append([0])
    EmissionSum.append([0])
    xlabel_ex.append([0])
    Station.append([0])

# set the frist value of SOC
for i in range(Number_EVs):
    SOC_List[i] = [SOC_Start[i]]
    SOC_cache[i] = SOC_Start[i]

# if SimConfig.Level == 1 or SimConfig.Forecast_current == 'no':
#    data_SP = pd.read_csv('./prognose/Eingangsdaten/Strompreis_Prognose_Mittelwert.csv', sep=';', index_col=0)
# data_SP = [26.807]*96
# else:
# read the current price csv
data_SP = pd.read_csv('./prognose/Eingangsdaten/Strompreis_final_test.csv', sep=';', index_col=0, parse_dates=True)
# read the building loads csv
data_GL = pd.read_csv('./prognose/Eingangsdaten/Gebaeudelast_1819_final.csv', sep=';', index_col=0, parse_dates=True)
# read the PV csv
data_PV = pd.read_csv('./prognose/Eingangsdaten/Globalstrahlung_final.csv', sep=';', index_col=0, parse_dates=True)
# data_PV = pd.read_csv('./prognose/Eingangsdaten/Globalstrahlung_final_zero.csv', sep= ';', index_col=0, parse_dates=True)
# read the energie source csv
# data_EQ = pd.read_csv('./prognose/Eingangsdaten/Energiequelle.csv', sep= ';', index_col=0, parse_dates=True)
data_EQ = pd.read_csv('./prognose/Eingangsdaten/CO2Emissionen_final_test.csv', sep=';', index_col=0, parse_dates=True)
# read the prediction begin
simulation_begin = SimConfig.start_timestamp
# transform into time form
simulation_begin_localtime = time.localtime(simulation_begin)
simulation_begin_time = time.strftime("%d.%m.%Y %H:%M", simulation_begin_localtime)
simulation_begin_H = time.strftime("%H", simulation_begin_localtime)

# calculate the result of grid - charging pwr of ked over the full simulation duration
# saving them into a df, which will be access later
start_date = SimConfig.time
duration = SimConfig.SimDuration * 96 +forecast_out
date_range = pd.date_range(start=start_date, periods=duration, freq='15 min')
full_dates = []
for dates in date_range:
    full_dates.append(datetime.datetime.timestamp(dates))
df_grid = {'Timestamp': full_dates}
df_grid = pd.DataFrame(df_grid)
grid_with_ked = []
for ts in full_dates:
    ked_now = get_current_ked_number(ts)
    grid_now = SimConfig.Grid - ked_now * SimConfig.CP_KED
    grid_with_ked.append(grid_now)
df_grid['Grid'] = grid_with_ked

# run the 1st prediction of the 1st package
P1p = Prognose1p.Prognosis1P()
P1p.run(simulation_begin_time, simulation_begin_localtime, forecast_list=['SP', 'PV', 'GL', 'CO2'])

# set up all the data for all EVs
for x in range(1, Number_EVs + 1):
    # added:
    # new absolute soc diff (SOC_to_charge = SOC_diff_for_next_route + SOC_buffer - SOC_Now)
    data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")
    # added:
    # new absolute soc diff (SOC_to_charge = SOC_diff_for_next_route + SOC_buffer - SOC_Now)
    print('EV:', x, 'SOC NOW: ', SOC_Now[x - 1])
    try:
        first_soc_prog = int(
            np.where(np.logical_and(~data_CP[str('SOC_') + str(x)].isnull(), data_CP[str('SOC_') + str(x)] != 0))[0][0])
        data_CP.loc[first_soc_prog, str('SOC_') + str(x)] = data_CP.loc[first_soc_prog, str('SOC_') + str(x)] \
                                                            + SimConfig.SOC_buffer - (SOC_Now[x - 1] / 100)
    except:
        pass

    # first_soc_prog = int(np.where(~data_CP[str('SOC_') + str(x)].isnull())[0][0])
    # data_CP.loc[first_soc_prog, str('SOC_') + str(x)] = data_CP.loc[first_soc_prog, str('SOC_') + str(x)] \
    #                                                     + SimConfig.SOC_buffer - (SOC_Now[x - 1] / 100)

    # data_CP.loc[~data_CP[str('SOC_') + str(x)].isnull( ), str('SOC_') + str(x)] = data_CP[str('SOC_') + str(x)] \
    #                                                                               + SimConfig.SOC_buffer - (
    #                                                                                           SOC_Now[x - 1] / 100)
    data_CP.loc[data_CP[str('Charging_') + str(x)] == 0, str('SOC_') + str(x)] = 0
    data_CP.loc[data_CP[str('SOC_') + str(x)] < 0, str('SOC_') + str(x)] = 0.001
        #
    data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", header=True, index=False)
    # added end

    # read test scenario files of every vehicle
    df = pd.read_csv('./TestScenario/TestScenario' + str(x) + '.csv')
    # sort the charging end column
    TS = np.array(df['LadeEnde'])
    # search the nearest charging end before simulation begin
    Index_begin[x] = np.searchsorted(TS, [simulation_begin, ], side='left')[0]
    Index_begin[x] = Index_begin[x] - 1
    # run the 1st prediction of the 2nd package of vehicles
    P2p = Prognose2p.Prognosis2P()
    print(x)
    # print(df.at[Index_begin[x]-1,'Ladezeit'])
    print(df.at[Index_begin[x] - 1, 'Fahrzeit'])
    print(df.at[Index_begin[x] - 1, 'Verbrauch_SoC'])
    P2p.predict_values(df.at[Index_begin[x] - 1, 'Ladezeit'], df.at[Index_begin[x] - 1, 'Fahrzeit'],
                       df.at[Index_begin[x] - 1, 'Verbrauch_SoC'], x)
    # read the result of prediction of the 2nd package of vehicles
    df1 = pd.read_csv("./prognose2p/results/Prognose_EV_" + str(x) + ".csv")
    # prognosis of the vehicles from last charging event
    prognosis_begin_otherevs = df.at[Index_begin[x], 'LadeEnde']
    CS.append(prognosis_begin_otherevs)
    # buildup of the charging csv with the beginning as end of charging
    GFAC = GestaltungFahrplanAfterCharging.GestaltungFahrplan()
    GFAC.Gestaltung(read_path="./prognose2p/results/Prognose_EV_" + str(x) + ".csv", \
                    to_path="./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", \
                    duration=int((simulation_begin - prognosis_begin_otherevs) // 900 + 96), \
                    ID=x)

    # selection of the last 96 data which shares the same time as the target vehicle
    df2 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")
    df3 = df2[-96:]
    df3.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", index=False)
    df3 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")

    # if the prediction result shows the vehicle is not being charged
    # but the vehicle is in reality being charged
    if df3.at[0, str('Charging_') + str(x)] == '' and df.at[Index_begin[x] + 1, 'LadeStart'] < simulation_begin:
        # locate the last charging start before simulation beginning
        real_chargingstart = df.at[Index_begin[x] + 1, 'LadeStart']
        # calculate the diffence of simulation beginning and last charging start
        diff_chargingstart = (simulation_begin - real_chargingstart) // 900
        # run the 1st prediction of the 2nd package of vehicles
        P2p = Prognose2p.Prognosis2P()
        P2p.predict_values(df.at[Index_begin[x], 'Ladezeit'], \
                           df.at[Index_begin[x], 'Fahrzeit'], \
                           df.at[Index_begin[x], 'Verbrauch_SoC'], x)
        # build up of the charging plan from the beginning of last charging start
        GF = GestaltungFahrplan.GestaltungFahrplan()
        GF.Gestaltung(read_path="./prognose2p/results/Prognose_EV_" + str(x) + ".csv", \
                      to_path="./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", \
                      duration=diff_chargingstart + 96)
        # sort the last 96 rows
        df2 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")
        df3 = df2[-96:]
        df3.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", index=False)
        df3 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")


    # if the prediction shows the vehicle is being charged
    # and the vehicle is in reality not being charged
    elif df3.at[0, str('Charging_') + str(x)] == 1 and df.at[Index_begin[x] + 1, 'LadeStart'] > simulation_begin + 900:
        # sort the =SOC_to_charge in last row
        SOC_storage = df3.at[95, str('SOC_to_charge_') + str(x)]
        # insert a row of empty values
        df0 = pd.DataFrame([['', '']], columns=[str('Charging_') + str(x), str('SOC_to_charge_') + str(x)])
        df0.append(df3, ignore_index=True)
        df3 = df0[-96:]
        df3.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", index=False)
        df3 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")
        # adjust the last charging value after inserting an empty row into the 1st row
        if SOC_storage != '':
            Charging_storage = 0
            for i in range(96):
                if df3.at[95 - i, str('Charging_') + str(x)] == 1:
                    Charging_storage = Charging_storage + 1
                else:
                    break
            df3.at[95, str('SOC_to_charge_') + str(x)] = SOC_storage * Charging_storage / (Charging_storage + 1)
            df3.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", index=False)
            df3 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv")

    if df.at[Index_begin[x] + 1, 'LadeStart'] <= simulation_begin and df.at[
        Index_begin[x] + 1, 'LadeEnde'] > simulation_begin:
        Station[x - 1] = [1]
    else:
        Station[x - 1] = [0]

    # judge if the required SOC is chargeable
    if df3.at[0, str('SOC_') + str(x)] > P_max / 4 / E_EV:
        SOC_not_chargable = df3.at[0, str('SOC_') + str(x)] - P_max / 4 / E_EV
        df3.at[0, str('SOC_') + str(x)] = P_max / 4 / E_EV
        df3.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(x) + ".csv", index=False)

    SOC_to_charge_sum = SOC_to_charge_sum + df3.at[0, str('SOC_') + str(x)]

    if df3.at[0, str('SOC_') + str(x)] > 0:
        EV_charging_ID.append(x)

# binding of all data
# adapt of the grid data between 18:00 and 7:00 (obsolete)
df = pd.read_csv("Prognose_1P.csv")

timestamp_first_row = df.at[0, 'TimeIndex']
time_local = time.localtime(timestamp_first_row)
time_check = time.strftime("%d.%m.%Y %H:%M", time_local)
time_check_index_PV = np.where(data_PV.index == time_check)[0]
time_check_index_PV = int(time_check_index_PV)
# transform the data from global radiation into the power for checking
real_PV_Globalstrahlung = data_PV.iat[time_check_index_PV, 3]
PV_cos = math.cos(math.pi * SimConfig.Illumination_Angle / 180)
PV_power_check = real_PV_Globalstrahlung * PV_cos * 0.01 * SimConfig.Efficiency_PV * SimConfig.Area_PV * 0.001
time_check_index_GL = np.where(data_GL.index == time_check)[0]
time_check_index_GL = int(time_check_index_GL)
time_check_index_SP = np.where(data_SP.index == time_check)[0]
time_check_index_SP = int(time_check_index_SP)
time_check_index_EQ = np.where(data_EQ.index == time_check)[0]
time_check_index_EQ = int(time_check_index_EQ)
df.at[0, 'PVPower'] = PV_power_check
df.at[0, 'Gebaeudelast'] = data_GL.iat[time_check_index_GL, 2]
# TODO Wenn Simulation Test zuende dann ## der nächsten zwei Zeilen raus nehmen
df.at[0, 'Strompreis'] = data_SP.iat[time_check_index_SP, 2]
df.at[0, 'CO2'] = data_EQ.iat[time_check_index_EQ, 3]
for i in range(int(df.shape[0])):
    timestamp = df.at[i, 'TimeIndex']
    time_local = time.localtime(timestamp)
    row = int(np.where(df_grid['Timestamp'] == timestamp)[0])
    grid_value = df_grid.loc[row, 'Grid']
    df.at[i, 'Grid'] = grid_value
    # number_ked_now = get_current_ked_number(timestamp)
    # df.at[i, 'Grid'] = SimConfig.Grid - CP_KED * number_ked_now
    # time_H = time.strftime("%H",time_local)
    # time_H = int(time_H)
    # if time_H in range(0,7) or time_H in range(18,24):
    #     df.at[i,'Grid'] = SimConfig.Grid - CP_KED * Number_KED #todo ked
df.to_csv("Prognose_1P.csv", header=True, index=False)

for i in range(1, Number_EVs + 1):
    df2 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
    col_name1 = df.columns.tolist()
    col_name2_1 = str('Charging_') + str(i)
    col_name2_2 = str('SOC_to_charge_') + str(i)

    col_name1.append(col_name2_1)
    col_name1.append(col_name2_2)

    df = df.reindex(columns=col_name1)

    df[str('Charging_') + str(i)] = df2[str('Charging_') + str(i)]
    df[str('SOC_to_charge_') + str(i)] = df2[str('SOC_') + str(i)]

    df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

for i in range(1, Number_EVs + 1):
    if df.at[0, 'Charging_' + str(i)] == 1:
        EV_charging_ID_1.append(i)
print(EV_charging_ID_1)

c = 0
len_index_sum = 0
for i in EV_charging_ID_1:
    b = df['SOC_to_charge_' + str(i)].tolist()
    # locate current SOC to charge
    a = next(x for x in b if x >  # 'TODO frage > =
             0)
    index_SOC = b.index(a)  # index in liste wo geg ev soc am nächsten lädt
    if a * E_EV / (index_SOC + 1) > P_max / 4:  # wenn soc*cap/anzahl intervalle (=ladeleistung in 15 min)
        df.at[int(index_SOC), 'SOC_to_charge_' + str(i)] = P_max / 4 * (index_SOC + 1) / E_EV
        df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)
    c = c + a / (index_SOC + 1)  # summe der durchsschnt soc pro intervall
    len_index_sum = len_index_sum + index_SOC + 1
    index_SOC_1[i] = index_SOC
    SOC_1[i] = a
    SOC_average_1[i] = a / (index_SOC + 1)

# wenn summe der durchschnittlichen ladeleistung pro intervall größer als
if c * E_EV * 4 > df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']:
    Power_available = df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']
    Power_to_assign = Power_available
    Power_percent = Power_available / (c * E_EV * 4)
    for i in EV_charging_ID_1:
        df.at[int(index_SOC_1[i]), 'SOC_to_charge_' + str(i)] = Power_percent * SOC_1[i]
    df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

if SOC_to_charge_sum * E_EV > df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']:
    Power_available = df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']
    Power_to_assign = Power_available
    Power_average = Power_available / len(EV_charging_ID)
    for i in range(1, Number_EVs + 1):
        if df.at[0, str('SOC_to_charge_') + str(i)] < Power_average:
            Power_to_assign = Power_to_assign - df.at[0, str('SOC_to_charge_') + str(i)]
            EV_charging_ID.remove(i)
        else:
            continue
    for i in EV_charging_ID:
        df.at[0, str('SOC_to_charge_') + str(i)] = Power_to_assign / len(EV_charging_ID)

    df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

    EV_charging_ID_1.clear()
    EV_charging_ID.clear()
    SOC_to_charge_sum = 0

time_check = time.strftime("%d.%m.%Y %H:%M", time_local)
time_check_index_PV = np.where(data_PV.index == time_check)[0]
time_check_index_PV = int(time_check_index_PV)
# transform the data from global radiation into the power for checking
real_PV_Globalstrahlung = data_PV.iat[time_check_index_PV, 3]
PV_cos = math.cos(math.pi * SimConfig.Illumination_Angle / 180)
PV_power_check = real_PV_Globalstrahlung * PV_cos * 0.01 * SimConfig.Efficiency_PV * SimConfig.Area_PV * 0.001
time_check_index_GL = np.where(data_GL.index == time_check)[0]
time_check_index_GL = int(time_check_index_GL)

# Load_limit.append(df.at[0,'Grid']+df.at[0,'PVPower']-df.at[0,'Gebaeudelast'])

Load_limit.append(df.at[0, 'Grid'] + PV_power_check - int(data_GL.iat[time_check_index_GL, 2]))
load_limit_PV.append(PV_power_check)
load_limit_GL.append(data_GL.iat[time_check_index_GL, 2])
load_limit_KED.append(SimConfig.Grid - df.at[0, 'Grid'])
# TODO ausgabe
for i in range(15):
    y_LL_stacked.append(Load_limit[0])

# #TODO studentierisung
# df = pd.read_csv("./prognose/Prognose_final.csv")
# df['Strompreis'], df['CO2'] = studentisierung(df['Strompreis'], df['CO2'])
# df.to_csv("./prognose/Prognose_final.csv",header = True, index = False)

# running of the optimization algorithm
LK = Optimization.Optimierung()

# storage the time for checking if the prediction result is accurate enough for later
df1P = pd.read_csv("Prognose_1P.csv")
timestamp_check = df1P.at[0, 'TimeIndex']
time_local = time.localtime(timestamp_check)
time_check = time.strftime("%d.%m.%Y %H:%M", time_local)
# set the 1st xlabel
xlabel.append(time_check)
time_check_index_SP = np.where(data_SP.index == time_check)[0]
time_check_index_SP = int(time_check_index_SP)
# if SimConfig.Level == 1 or SimConfig.Forecast_current == 'no':
#     #Price_Now = 26.807
#     day = time_local.tm_mday
#     hour = time_local.tm_hour
#     minute = time_local.tm_min
#     row = (day - 1) * 96 + hour * 4 + minute / 15
#     row = int(row)
#     df_price = pd.read_csv('./prognose/Eingangsdaten/Strompreis_Prognose_Mittelwert.csv', sep=';', index_col=0)
#     Price_Now = df_price.iloc[row, 0]
# else:
# find the current price in reality
# real price for plotting etc.
Price_Now = data_SP.iat[time_check_index_SP, 2]
# find the energie source in reality
time_check_index_EQ = np.where(data_EQ.index == time_check)[0]
time_check_index_EQ = int(time_check_index_EQ)
# if SimConfig.Level == 1 and SimConfig.Forecast_CO2 == 'no':
#     day = time_local.tm_mday
#     hour=time_local.tm_hour
#     minute=time_local.tm_min
#     row = (day-1)*96+hour*4+minute/15
#     row = int(row)
#     df = pd.read_excel('./prognose/Eingangsdaten/200605_Statistik_Emissionen.xlsx')
#     CO2_Now = df.iloc[row,2]/3600 #überprüfen welche daten sind daS?
# else:
# #calculate the CO2 Emission in reality
# CO2_Now = (data_EQ.iat[time_check_index_EQ,4]*FaktorBK+data_EQ.iat[time_check_index_EQ\
#                      ,5]*FaktorSK+data_EQ.iat[time_check_index_EQ,6]*FaktorEG+data_EQ.\
#             iat[time_check_index_EQ,7]*FaktorSon)/data_EQ.iat[time_check_index_EQ,8]
# read out the real co2 emission
# CO2_Now = data_EQ.iat[time_check_index_EQ, 3]
CO2_Now = data_EQ.iat[time_check_index_EQ, 3]
# calculate load limit

# TODO inti ohne opt
# if optimazation isnt required:
if SimConfig.No_Opt == 1:
    # get load limit
    load_limit = Load_limit[-1]
    # count the charging evs, only the first row is required
    if SimConfig.Static_Algo == 'EQUAL':
        list_of_charging_evs_for_equal = []
        for EV in range(1, Number_EVs + 1):
            charging_state = df.at[0, 'Charging_%s' % EV]
            if charging_state == 1 and SOC_Now[EV - 1] < 100:
                list_of_charging_evs_for_equal.append(EV)
        charging_power_equal = LK.equal(len(list_of_charging_evs_for_equal), load_limit)
    else:
        for EV in range(1, 6 + 1):
            charging_state = df.at[0, 'Charging_%s' % EV]
            # check if EV is at the station and the SOC is under 100
            if charging_state == 1 and SOC_Now[EV - 1] < 100:
                # check if the EV is already charging
                tuple = [item for item in list_charging_evs_with_prio if item[0] == EV]
                # the EV is already in the last list
                if tuple:
                    # counting up priority
                    pos = list_charging_evs_with_prio.index(tuple[0])
                    new_prio = tuple[0][1] + 1
                    list_charging_evs_with_prio[pos] = (EV, new_prio)
                # EV starts charging, add to the list
                else:
                    list_charging_evs_with_prio.append((EV, 0))
            # EV left/is on a trip
            else:
                tuple = [item for item in list_charging_evs_with_prio if item[0] == EV]
                # remove EV from list if it was in the list
                if tuple:
                    list_charging_evs_with_prio.remove(tuple[0])
        charging_power_fcfs_list = LK.fcfs(load_limit, list_charging_evs_with_prio)
else:
    ladeplan_opt = LK.ladeplan(Number_EVs)

for i in range(Number_EVs):
    # storage of outputs of the algorith,
    if SimConfig.No_Opt == 1:
        if SimConfig.Static_Algo == 'EQUAL':
            if i + 1 in list_of_charging_evs_for_equal:
                power_1st[i] = charging_power_equal
            else:
                power_1st[i] = 0
        else:
            power_1st[i] = charging_power_fcfs_list[i]

    else:
        power_1st[i] = ladeplan_opt[i]  # LK.ladeplan(Number_EVs)[i]

    # check if SOC will be bigger than 100
    print(power_1st[i])
    if SOC_Start[i] + power_1st[i] / 4 / E_EV * 100 <= 100:
        # calculation of the charged power
        chargingpower[i] = chargingpower[i] + power_1st[i]
        # buildup of the list of chargingpower
        ChargingPower[i] = [power_1st[i]]
        # buildup of the list of chargingpower for stackplot
        y_CP_stacked[i] = [power_1st[i]]
        for k in range(14):
            y_CP_stacked[i].append(power_1st[i])
        # calculate the current SOC
        SOC_Now[i] = SOC_Start[i] + power_1st[i] / 4 / E_EV * 100
        # buildup of SOC_List
        SOC_List[i].append(SOC_Now[i])
        # buildup of price list
        Price[i].append(power_1st[i] * Price_Now / 4)
        # buildup of pricesum list
        PriceSum[i].append(Price[i][-1] + PriceSum[i][-1])
        # buildup of CO2_List
        Emission[i].append(power_1st[i] * CO2_Now / 4)
        # buildup of emissionsum list
        EmissionSum[i].append(Emission[i][-1] + EmissionSum[i][-1])
    else:
        # calculate the chargable SOC
        SOC_Diff = 100 - SOC_Start[i]
        # calculate the chargable power
        power_1st[i] = SOC_Diff * E_EV * 0.04
        # calculation of the charged power
        chargingpower[i] = chargingpower[i] + power_1st[i]
        # buildup of the list of chargingpower
        ChargingPower[i] = [power_1st[i]]
        # buildup of the list of chargingpower for stackplot
        y_CP_stacked[i] = power_1st[i]
        for k in range(14):
            y_CP_stacked[i].append(power_1st[i])
        # calculate the current SOC
        SOC_Now[i] = 100
        # buildup of SOC_List
        SOC_List[i].append(SOC_Now[i])
        # buildup of price list
        Price[i].append(power_1st[i] * Price_Now / 4)
        # buildup of pricesum list
        PriceSum[i].append(Price[i][-1] + PriceSum[i][-1])
        # buildup of CO2_List
        Emission[i].append(power_1st[i] * CO2_Now / 4)
        # buildup of emissionsum list
        EmissionSum[i].append(Emission[i][-1] + EmissionSum[i][-1])
    df_TS = pd.read_csv('./TestScenario/TestScenario' + str(i + 1) + '.csv')
    if df_TS.at[Index_begin[i + 1], 'LadeEnde'] < simulation_begin < df_TS.at[Index_begin[i + 1] + 1, 'LadeStart']:
        driving_before = df_TS.at[Index_begin[i + 1], 'LadeEnde']
        driving_after = df_TS.at[Index_begin[i + 1] + 1, 'LadeStart']
        driving_SOC = df_TS.at[Index_begin[i + 1], 'Verbrauch_SoC']
        SOC_before[i] = 70 + (simulation_begin - driving_before) / (driving_after - driving_before) * driving_SOC
        SOC_cache[i] = SOC_before[i]
print(SOC_cache)
# if simulation_begin + 900 > df_TS.at[int(Index_begin[i+1]+1),'LadeStart']:
# SOC_List[i][-1] = SOC_List[i][-1]-df_TS.at[int(Index_begin[i+1]),'Verbrauch_SoC']
# read the simulation duration from simulation configuration
SimDuration = SimConfig.SimDuration

# calculate the SOC_cache for EVs which are driven away at the simulation beginning


# loop the program for the set duration
for j in range(1, int(SimDuration * 96)):
    # judge if the prediction of 1st package is right
    df1P = pd.read_csv("Prognose_1P.csv")
    # read the checking timestamp from prediction
    timestamp_check = df1P.at[0, 'TimeIndex']
    # set the timestamp into another form
    time_local = time.localtime(timestamp_check)
    time_check = time.strftime("%d.%m.%Y %H:%M", time_local)
    # if SimConfig.Level == 1 or SimConfig.Forecast_current == 'no':
    #     #Price_Now = 26.807
    #     day = time_local.tm_mday
    #     hour = time_local.tm_hour
    #     minute = time_local.tm_min
    #     row = (day - 1) * 96 + hour * 4 + minute / 15
    #     row = int(row)
    #     df_price = pd.read_csv('./prognose/Eingangsdaten/Strompreis_Prognose_Mittelwert.csv', sep=';', index_col=0)
    #     Price_Now = df_price.iloc[row, 0]
    # else:
    # locate the checking time in real current price csv
    time_check_index_SP = np.where(data_SP.index == time_check)[0]
    time_check_index_SP = int(time_check_index_SP)
    # locate the checking time in real building load csv
    time_check_index_GL = np.where(data_GL.index == time_check)[0]
    time_check_index_GL = int(time_check_index_GL)
    # locate the checking time in real PV csv
    time_check_index_PV = np.where(data_PV.index == time_check)[0]
    time_check_index_PV = int(time_check_index_PV)
    # locate the checking time in Energie source csv
    time_check_index_EQ = np.where(data_EQ.index == time_check)[0]
    time_check_index_EQ = int(time_check_index_EQ)
    # transform the data from global radiation into the power for checking
    real_PV_Globalstrahlung = data_PV.iat[time_check_index_PV, 3]
    # print('real PV:', real_PV_Globalstrahlung, 'timecheck: ', time_check_index_PV)
    print('TIMESTAMP:', time_check)
    PV_cos = math.cos(math.pi * SimConfig.Illumination_Angle / 180)
    PV_power_check = real_PV_Globalstrahlung * PV_cos * 0.01 * SimConfig.Efficiency_PV * SimConfig.Area_PV * 0.001
    if SimConfig.Level > 1:
        # if difference is greater then 20% predict again
        # todo ändern wenn Prognoseabweichung ändern
        if abs(df1P.at[0, 'PVPower'] - PV_power_check) >= 0.2 * (PV_power_check + 1) or \
                abs(df1P.at[0, 'CO2'] - data_EQ.iat[time_check_index_EQ, 3]) >= 0.2 * data_EQ.iat[
            time_check_index_EQ, 3] or \
                abs(df1P.at[0, 'Gebaeudelast'] - data_GL.iat[time_check_index_GL, 2]) >= 0.2 * data_GL.iat[
            time_check_index_GL, 2] or \
                abs(df1P.at[0, 'Strompreis'] - data_SP.iat[time_check_index_SP, 2]) >= 0.2 * data_SP.iat[
            time_check_index_SP, 2]:
            ''' pv globalstrahlung real in col 3,
                             strompreis in col 2
                             co2 col 3'''
            print('############ Neue Prognose nötig ###############')
            new_forecast_needed = []
            if abs(df1P.at[0, 'Gebaeudelast'] - data_GL.iat[time_check_index_GL, 2]) >= 0.2 * data_GL.iat[
                time_check_index_GL, 2]:
                print(df1P.at[0, 'Gebaeudelast'], 'datagl ', data_GL.iat[time_check_index_GL, 2])
                print('### Gebäudelast ##')  #
                new_forecast_needed.append('GL')
            if abs(df1P.at[0, 'Strompreis'] - data_SP.iat[time_check_index_SP, 2]) >= 0.2 * data_SP.iat[
                time_check_index_SP, 2]:
                print('## Strompreis ##')
                print('Überprüfen pred: ', df1P.at[0, 'Strompreis'], 'real: ', data_SP.iat[time_check_index_SP, 2])
                new_forecast_needed.append('SP')
            if abs(df1P.at[0, 'PVPower'] - PV_power_check) >= 0.2 * PV_power_check + 1:
                print('## pv ## ')
                print('pred: ', df1P.at[0, 'PVPower'], 'power check: ', PV_power_check)
                new_forecast_needed.append('PV')
            if abs(df1P.at[0, 'CO2'] - data_EQ.iat[time_check_index_EQ, 3]) >= 0.2 * data_EQ.iat[
                time_check_index_EQ, 3]:
                print('## CO2 ##')
                print('Überprüfen pred: ', df1P.at[0, 'CO2'], 'real: ', data_EQ.iat[time_check_index_EQ, 3])
                new_forecast_needed.append('CO2')

            # predict the 1st package again
            # drop the last prediction result of last 15 minutes
            df1P = df1P.drop([0])
            df1P.to_csv("Prognose_1P.csv", header=True, index=False)
            df1P = pd.read_csv("Prognose_1P.csv")
            # timestamp of prediction begin
            prognosis_begin = df1P.at[0, 'TimeIndex']
            # transform into time form
            prognosis_begin_localtime = time.localtime(prognosis_begin)
            prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M", prognosis_begin_localtime)
            print('Timestamp::::', prognosis_begin_time)
            # run the 1st prediction of the 1st package
            P1p.run(prognosis_begin_time, prognosis_begin_localtime, forecast_list=new_forecast_needed)
        else:
            # no need to predic the 1st package again
            df1P = df1P.drop([0])
            df1P.to_csv("Prognose_1P.csv", header=True, index=False)
            df1P = pd.read_csv("Prognose_1P.csv")
    else:
        df1P = df1P.drop([0])
        df1P.to_csv("Prognose_1P.csv", header=True, index=False)
        df1P = pd.read_csv("Prognose_1P.csv")
        # timestamp of prediction begin
        prognosis_begin = df1P.at[0, 'TimeIndex']
        # transform into time form
        prognosis_begin_localtime = time.localtime(prognosis_begin)
        prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M", prognosis_begin_localtime)
        # run the 1st prediction of the 1st package
        P1p.run(prognosis_begin_time, prognosis_begin_localtime, forecast_list=[])

    # if OTM_Target == 'Cost':
    #     if SimConfig.Level > 1:
    #         #if the difference is greater than 20%
    #         #if abs(df1P.at[0,'Strompreis']-data_SP.iat[time_check_index_SP,0]) >= 0.2*data_SP.iat[time_check_index_SP,0] or \
    #         #abs(df1P.at[0,'Gebaeudelast']-data_GL.iat[time_check_index_GL,0]) >= 0.2*data_SP.iat[time_check_index_GL,0] or \
    #         #abs(df1P.at[0,'PVPower']-PV_power_check) >= 0.2*PV_power_check:
    #         #if #abs(df1P.at[0,'Strompreis']-data_SP.iat[time_check_index_SP,0]) >= 0.2*data_SP.iat[time_check_index_SP,0] or\
    #            # abs(df1P.at[0,'Gebaeudelast']-data_GL.iat[time_check_index_GL,1]) >= 0.2*data_GL.iat[time_check_index_GL,1] or \
    #         if   abs(df1P.at[0,'PVPower']-PV_power_check) >= 0.2*(PV_power_check+1) or \
    #             abs(df1P.at[0, 'CO2'] - data_EQ.iat[time_check_index_EQ, 3]) >= 0.2 * data_EQ.iat[time_check_index_EQ, 3]:
    #             ''' pv globalstrahlung real in col 3,
    #              strompreis in col 2
    #              co2 col 3'''
    #             print('#################### Neue Prognose nötig ###############')
    #             new_forecast_needed = []
    #             # Note: vorher sonst Profilwert ausgelesen
    #             if abs(df1P.at[0,'Gebaeudelast']-data_GL.iat[time_check_index_GL,1]) >= 0.2*data_GL.iat[time_check_index_GL,1]:
    #                 print(df1P.at[0,'Gebaeudelast'], 'datagl ', data_GL.iat[time_check_index_GL,1])
    #                 print(time_check_index_GL)
    #                 print('### Gebäudelast ##')#
    #                 new_forecast_needed.append('GL')
    #             if abs(df1P.at[0,'Strompreis']-data_SP.iat[time_check_index_SP,2]) >= 0.2*data_SP.iat[time_check_index_SP,2]:
    #                 print('## Strompreis ##')
    #                 print('Überprüfen pred: ', df1P.at[0,'Strompreis'], 'real: ', data_SP.iat[time_check_index_SP,2])
    #                 new_forecast_needed.append('SP')
    #             if abs(df1P.at[0,'PVPower']-PV_power_check) >= 0.2*PV_power_check+1:
    #                 print('## pv ## ')
    #                 print('pred: ',df1P.at[0,'PVPower'], 'power check: ', PV_power_check)
    #                 new_forecast_needed.append('PV')
    #             if abs(df1P.at[0, 'CO2'] - data_EQ.iat[time_check_index_EQ, 3]) >= 0.2 * data_EQ.iat[time_check_index_EQ, 3]:
    #                 print('## CO2 ##')
    #                 print('Überprüfen pred: ', df1P.at[0, 'CO2'], 'real: ', data_EQ.iat[time_check_index_EQ, 3])
    #                 new_forecast_needed.append('CO2')
    #             # predict the 1st package again
    #             # drop the last prediction result of last 15 minutes
    #             df1P = df1P.drop([0])
    #             df1P.to_csv("Prognose_1P.csv",header = True, index = False)
    #             df1P = pd.read_csv("Prognose_1P.csv")
    #             #timestamp of prediction begin
    #             prognosis_begin = df1P.at[0,'TimeIndex']
    #             #transform into time form
    #             prognosis_begin_localtime = time.localtime(prognosis_begin)
    #             prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M",prognosis_begin_localtime)
    #             print('Timestamp::::', prognosis_begin_time)
    #             #run the 1st prediction of the 1st package
    #             #P1p = Prognose1p.Prognosis1P()
    #             P1p.run(prognosis_begin_time,prognosis_begin_localtime, forecast_list=new_forecast_needed)
    #         else:
    #             #no need to predic the 1st package again
    #             df1P = df1P.drop([0])
    #             df1P.to_csv("Prognose_1P.csv",header = True, index = False)
    #             df1P = pd.read_csv("Prognose_1P.csv")
    #     else:
    #         df1P = df1P.drop([0])
    #         df1P.to_csv("Prognose_1P.csv",header = True, index = False)
    #         df1P = pd.read_csv("Prognose_1P.csv")
    #         #timestamp of prediction begin
    #         prognosis_begin = df1P.at[0,'TimeIndex']
    #         #transform into time form
    #         prognosis_begin_localtime = time.localtime(prognosis_begin)
    #         prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M",prognosis_begin_localtime)
    #         #run the 1st prediction of the 1st package
    #        # P1p = Prognose1p.Prognosis1P()
    #        # P1p.run(prognosis_begin_time,prognosis_begin_localtime)
    # else:
    #     if SimConfig.Level > 1:
    #         # CO2_check = (data_EQ.iat[time_check_index_EQ,4]*FaktorBK+data_EQ.iat[time_check_index_EQ\
    #         #              ,5]*FaktorSK+data_EQ.iat[time_check_index_EQ,6]*FaktorEG+data_EQ.\
    #         #     iat[time_check_index_EQ,7]*FaktorSon)/data_EQ.iat[time_check_index_EQ,8]
    #         CO2_check = data_EQ.iat[time_check_index_EQ, 3]
    #         #if the difference is greater than 20%
    #         if abs(df1P.at[0,'CO2']-CO2_check) >= CO2_check or \
    #         abs(df1P.at[0,'Gebaeudelast']-data_GL.iat[time_check_index_GL,0]) >= 0.2*data_SP.iat[time_check_index_GL,0] or \
    #         abs(df1P.at[0,'PVPower']-PV_power_check) >= 0.2*PV_power_check:
    #             #predict the 1st package again
    #             #drop the last prediction result of last 15 minutes
    #             df1P = df1P.drop([0])
    #             df1P.to_csv("Prognose_1P.csv",header = True, index = False)
    #             df1P = pd.read_csv("Prognose_1P.csv")
    #             #timestamp of prediction begin
    #             prognosis_begin = df1P.at[0,'TimeIndex']
    #             #transform into time form
    #             prognosis_begin_localtime = time.localtime(prognosis_begin)
    #             prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M",prognosis_begin_localtime)
    #             #run the 1st prediction of the 1st package
    #            # P1p = Prognose1p.Prognosis1P()
    #             P1p.run(prognosis_begin_time,prognosis_begin_localtime, forecast_list = [])
    #         else:
    #             #no need to predic the 1st package again
    #             df1P = df1P.drop([0])
    #             df1P.to_csv("Prognose_1P.csv",header = True, index = False)
    #             df1P = pd.read_csv("Prognose_1P.csv")
    #     else:
    #         df1P = df1P.drop([0])
    #         df1P.to_csv("Prognose_1P.csv",header = True, index = False)
    #         df1P = pd.read_csv("Prognose_1P.csv")
    #         #timestamp of prediction begin
    #         prognosis_begin = df1P.at[0,'TimeIndex']
    #         #transform into time form
    #         prognosis_begin_localtime = time.localtime(prognosis_begin)
    #         prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M",prognosis_begin_localtime)
    #         #run the 1st prediction of the 1st package
    #         print('hier sollte ich nicht sein')
    #         P1p.run(prognosis_begin_time,prognosis_begin_localtime)
    # prepare the time_label for plotting
    timestamp_label = df1P.at[0, 'TimeIndex']
    time_label_local = time.localtime(timestamp_label)
    time_label = time.strftime("%d.%m.%Y %H:%M", time_label_local)

    for i in range(1, Number_EVs + 1):
        # read the test scenario file
        data_TS = pd.read_csv('./TestScenario/TestScenario' + str(i) + '.csv')
        data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
        # added:
        # new absolute soc diff (SOC_to_charge = SOC_diff_for_next_route + SOC_buffer - SOC_Now)
        # first_soc_prog = int(np.where(~data_CP[str('SOC_') + str(i)].isnull())[0][0])
        try:
            first_soc_prog = int(
                np.where(np.logical_and(~data_CP[str('SOC_') + str(i)].isnull(), data_CP[str('SOC_') + str(i)] != 0))[
                    0][0])
            data_CP.loc[first_soc_prog, str('SOC_') + str(i)] = data_CP.loc[first_soc_prog, str('SOC_') + str(i)] \
                                                                + SimConfig.SOC_buffer - (SOC_Now[i - 1] / 100)
        except:
            pass

        print('EV:', i, 'SOC NOW: ', SOC_Now[i - 1])
        # INFO: wegkommentier die zeile unter diesem kommi
        # data_CP.loc[~data_CP[str('SOC_') + str(i)].isnull(), str('SOC_') + str(i)] = data_CP[str('SOC_') + str(i)] \
        #                                                                              + SimConfig.SOC_buffer - (
        #                                                                                          SOC_Now[i - 1] / 100)
        data_CP.loc[data_CP[str('Charging_') + str(i)] == 0, str('SOC_') + str(i)] = 0
        data_CP.loc[data_CP[str('SOC_') + str(i)] < 0, str('SOC_') + str(i)] = 0.001
        # TODO sahdaihdashd nicht null weil sonst fehler stop itereation?
        #
        data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True, index=False)
        # added end

        # if the car is driven away
        if df1P.at[0, 'TimeIndex'] > data_TS.at[Index_begin[i], 'LadeEnde'] and \
                df1P.at[0, 'TimeIndex'] < data_TS.at[Index_begin[i] + 1, 'LadeStart']:
            # if simulation_begin < data_TS.at[Index_begin[i]+1,'LadeStart']:
            # SOC_cache[i-1] = SOC_before[i-1]

            Station[i - 1].append(0)

            if (df1P.at[0, 'TimeIndex'] < data_TS.at[Index_begin[i], 'LadeEnde'] + 900) or \
                    (df1P.at[0, 'TimeIndex'] - 1800 < data_TS.at[Index_begin[i], 'LadeStart'] and \
                     data_TS.at[Index_begin[i], 'LadeEnde'] < df1P.at[0, 'TimeIndex'] - 900):
                SOC_cache[i - 1] = SOC_Now[i - 1]
            data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
            # drop the prediction results of last 15 minutes
            data_CP = data_CP.drop([0])
            # set both charging and SOC_to_charge for the 1st row as 0
            toAdd = [0, 0]
            with open("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", "r") as infile:
                reader = list(csv.reader(infile))
                reader.insert(1, toAdd)

            with open("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", "w") as outfile:
                writer = csv.writer(outfile)
                for line in reader:
                    writer.writerow(line)
            data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
            col_name = data_CP.columns.tolist()
            data_CP = data_CP.reindex(columns=col_name)
            data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True, index=False)
            data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
            SOC_Now[i - 1] = SOC_Now[i - 1] - data_TS.at[Index_begin[i], 'Verbrauch_SoC'] / math.ceil(
                data_TS.at[Index_begin[i], 'Fahrzeit'])
            # for j in range(1,14):
            # if df1P.at[0,'TimeIndex']+900<data_TS.at[Index_begin[i]+1,'LadeEnde']:
            # SOC_Now[i-1] = SOC_Now[i-1] - data_TS.at[Index_begin[i],'Verbrauch_SoC']/math.ceil(data_TS.at[Index_begin[i],'Fahrzeit'])
            # SOC_cache[i-1] = SOC_Now[i-1]
        # if the car starts being charged
        elif df1P.at[0, 'TimeIndex'] >= data_TS.at[
            Index_begin[i] + 1, 'LadeStart']:  # and simulation_begin < data_TS.at[Index_begin[i]+1,'LadeStart']:
            # substract the SOC consumed during last driving
            # SOC_Now[i-1] = SOC_cache[i-1] - data_TS.at[Index_begin[i],'Verbrauch_SoC']

            Station[i - 1].append(1)

            if df1P.at[0, 'TimeIndex'] >= data_TS.at[Index_begin[i] + 1, 'LadeStart'] + 900:

                Index_begin[i] = Index_begin[i] + 1
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                data_CP = data_CP.drop([0])
                data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True,
                               index=False)
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                b = data_CP['SOC_' + str(i)].tolist()
                # locate current SOC to charge
                a = next(x for x in b if x > 0)  # 'TODO frage > =
                index_SOC = b.index(a)
                # if charged energy is enough according to the prediction result
                if data_CP.loc[index_SOC, 'SOC_' + str(i)] - (ChargingPower[i - 1][-1] / 4 / E_EV) < 0:
                    # set SOC for charging as 0
                    data_CP.loc[index_SOC, 'SOC_' + str(i)] = 0
                else:
                    # substract SOC charged from SOC for charging
                    data_CP.loc[index_SOC, 'SOC_' + str(i)] = data_CP.loc[index_SOC, 'SOC_' + str(i)] - (
                                ChargingPower[i - 1][-1] / 4 / E_EV)

                # storage the data
                data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True,
                               index=False)
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
            else:
                SOC_Now[i - 1] = SOC_cache[i - 1] - data_TS.at[Index_begin[i], 'Verbrauch_SoC']
                SOC_List[i - 1][-1] = SOC_Now[i - 1]
                Index_begin[i] = Index_begin[i] + 1
                # timestamp of charginging begin
                # charging_begin = data_TS.iat[Index_begin[x],2]
                # timestamp of prediction begin
                prognosis_begin = df1P.at[0, 'TimeIndex']
                # transform into time form
                prognosis_begin_localtime = time.localtime(prognosis_begin)
                prognosis_begin_time = time.strftime("%d.%m.%Y %H:%M", prognosis_begin_localtime)

                # run the 1st prediction of the 2nd package
                P2p = Prognose2p.Prognosis2P()
                P2p.predict_values(data_TS.at[Index_begin[i] - 1, 'Ladezeit'], \
                                   data_TS.at[Index_begin[i] - 1, 'Fahrzeit'], \
                                   chargingpower[i - 1] / E_EV * 100, i)
                # buildup of the charging plan
                GF = GestaltungFahrplan.GestaltungFahrplan()
                GF.Gestaltung(read_path="./prognose2p/results/Prognose_EV_" + str(i) + ".csv", \
                              to_path="./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", duration=forecast_out,
                              ID=i)
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                if i in EV_charging_ID:
                    b = data_CP['SOC_' + str(i)].tolist()
                    # locate current SOC to charge
                    a = next(x for x in b if x > 0)
                    index_SOC = b.index(a)
                    data_CP.loc[index_SOC, 'SOC_' + str(i)] = data_CP.loc[index_SOC, 'SOC_' + str(i)] + SOC_rest[i - 1]
        # elif df1P.at[0,'TimeIndex'] >= data_TS.at[Index_begin[i]+1,'LadeStart'] and simulation_begin >= data_TS.at[Index_begin[i]+1,'LadeStart']:

        # if car is in charge but there is no charging info according to the pronosis
        elif df1P.at[0, 'TimeIndex'] > data_TS.at[Index_begin[i], 'LadeStart'] and df1P.at[0, 'TimeIndex'] <= \
                data_TS.at[Index_begin[i], 'LadeEnde'] and data_CP.at[0, 'Charging_' + str(i)] == '':
            Station[i - 1].append(1)
            data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
            data_CP = data_CP.drop([0])
            data_CP.to_csv('./TestScenario/TestScenario' + str(i) + '.csv', header=True, index=False)
            data_CP = pd.read_csv('./TestScenario/TestScenario' + str(i) + '.csv')
        else:
            Station[i - 1].append(1)
            # return of the gained energy and gaining the left expected energy
            try:
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                data_CP = data_CP.drop([0])
                data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True,
                               index=False)
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                b = data_CP['SOC_' + str(i)].tolist()
                # locate current SOC to charge
                a = next(x for x in b if x > 0)  # 'TODO frage > =
                index_SOC = b.index(a)
                # if charged energy is enough according to the prediction result
                if data_CP.loc[index_SOC, 'SOC_' + str(i)] - (ChargingPower[i - 1][-1] / 4 / E_EV) < 0:
                    # set SOC for charging as 0
                    data_CP.loc[index_SOC, 'SOC_' + str(i)] = 0
                else:
                    # substract SOC charged from SOC for charging
                    data_CP.loc[index_SOC, 'SOC_' + str(i)] = data_CP.loc[index_SOC, 'SOC_' + str(i)] - (
                                ChargingPower[i - 1][-1] / 4 / E_EV)

                # storage the data
                data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True,
                               index=False)
                data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")


            except:
                # TODO? kurzer aufenthalt?
                # try:
                #     data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                #     data_CP = data_CP.drop([0])
                #     data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True,
                #                    index=False)
                #     data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
                # except:
                    Charging_rebuild = [1] * (forecast_out-1)
                    SOC_to_charge_rebuild = [0] * (forecast_out-1)
                    data_re = {'Charging_' + str(i): Charging_rebuild, 'SOC_' + str(i): SOC_to_charge_rebuild}
                    df_re = pd.DataFrame(data_re)
                    df_re.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", header=True,
                                 index=False)
                    data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")

        # if data_TS.at[Index_begin[i], 'LadeStart'] <= df1P.at[0,'TimeIndex'] and data_TS.at[
        #     Index_begin[i], 'LadeEnde'] > df1P.at[0,'TimeIndex']:
        #     Station[i - 1].append(1)
        # else:
        #     Station[i - 1].append(0)

        # judge if the required SOC is chargeable
        if data_CP.at[0, str('SOC_') + str(i)] > P_max / 4 / E_EV:
            SOC_not_chargable = data_CP.at[0, str('SOC_') + str(i)] - P_max / 4 / E_EV
            data_CP.at[0, str('SOC_') + str(i)] = P_max / 4 / E_EV
            data_CP.to_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv", index=False)

        data_CP = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
        s = data_CP.at[0, str('SOC_') + str(i)]
        if math.isnan(s) == True:
            s = 0
        SOC_to_charge_sum = SOC_to_charge_sum + s

        if data_CP.at[0, str('SOC_') + str(i)] > 0:
            EV_charging_ID.append(i)

    # binding all the data
    # adapt the grid data between 18:00 to 7:00
    df = pd.read_csv("Prognose_1P.csv")
    df.at[0, 'PVPower'] = PV_power_check
    df.at[0, 'Gebaeudelast'] = data_GL.iat[time_check_index_GL, 2]
    # TODO Wenn Simulationstest beendet  entfernen von # der nächsten zwei Zeilen
    df.at[0, 'Strompreis'] = data_SP.iat[time_check_index_SP, 2]
    df.at[0, 'CO2'] = data_EQ.iat[time_check_index_EQ, 3]

    for i in range(int(df.shape[0])):
        timestamp = df.at[i, 'TimeIndex']
        time_local = time.localtime(timestamp)
        row = int(np.where(df_grid['Timestamp'] == timestamp)[0])
        grid_value = df_grid.loc[row, 'Grid']
        df.at[i, 'Grid'] = grid_value
        # number_ked_now = get_current_ked_number(timestamp)
        # df.at[i, 'Grid'] = SimConfig.Grid - CP_KED * number_ked_now
        # time_H = time.strftime("%H",time_local)
        # time_H = int(time_H)
        # if time_H in range(0,7) or time_H in range(18,24):
        #     df.at[i,'Grid'] = SimConfig.Grid - CP_KED * Number_KED # todo ked einlesen
    df.to_csv("Prognose_1P.csv", header=True, index=False)

    for i in range(1, Number_EVs + 1):
        df2 = pd.read_csv("./prognose2p/charging_plan/chargingplan_EV_" + str(i) + ".csv")
        col_name1 = df.columns.tolist()
        col_name2_1 = str('Charging_') + str(i)
        col_name2_2 = str('SOC_to_charge_') + str(i)

        col_name1.append(col_name2_1)
        col_name1.append(col_name2_2)

        df = df.reindex(columns=col_name1)

        df[str('Charging_') + str(i)] = df2[str('Charging_') + str(i)]
        df[str('SOC_to_charge_') + str(i)] = df2[str('SOC_') + str(i)]
        # df[str('SOC_to_charge_') + str(i)] = df_SOC_absolute[:, ]

        df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)
    print(EV_charging_ID)
    if SOC_to_charge_sum * E_EV * 4 > df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']:
        Power_available = df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']
        Power_to_assign = Power_available
        Power_average = Power_available / len(EV_charging_ID)
        SOC_average = Power_average / 4 / E_EV
        for i in EV_charging_ID:
            if df.at[0, str('SOC_to_charge_') + str(i)] < SOC_average:
                Power_to_assign = Power_to_assign - df.at[0, str('SOC_to_charge_') + str(i)] * 4 * E_EV
                EV_charging_ID.remove(i)
            else:
                continue
        for i in EV_charging_ID:
            SOC_rest[i - 1] = df.at[0, str('SOC_to_charge_') + str(i)] - Power_to_assign / len(EV_charging_ID)
            df.at[0, str('SOC_to_charge_') + str(i)] = Power_to_assign / len(EV_charging_ID) / 4 / E_EV

        df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

    for i in range(1, Number_EVs + 1):
        if df.at[0, 'Charging_' + str(i)] == 1:
            EV_charging_ID_1.append(i)
    print(EV_charging_ID_1)

    for i in EV_charging_ID_1:
        a = 0
        for x in range(int(df.shape[0])):
            if df.at[int(x), 'SOC_to_charge_' + str(i)] > 0:
                a = a + 1
        if a == 0:
            EV_charging_ID_1.remove(int(i))

    c = 0
    len_index_sum = 0
    for i in EV_charging_ID_1:
        b = df['SOC_to_charge_' + str(i)].tolist()
        # locate current SOC to charge
        a = next(x for x in b if x > 0)  # 'TODO frage > =
        index_SOC = b.index(a)
        if a * E_EV / (index_SOC + 1) > P_max / 4:
            df.at[int(index_SOC), 'SOC_to_charge_' + str(i)] = P_max / 4 * (index_SOC + 1) / E_EV
            df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)
        c = c + a / (index_SOC + 1)
        len_index_sum = len_index_sum + index_SOC + 1
        index_SOC_1[i] = index_SOC
        SOC_1[i] = a
        SOC_average_1[i] = a / (index_SOC + 1)

    if c * E_EV * 4 > df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']:
        Power_available = df.at[0, 'Grid'] + df.at[0, 'PVPower'] - df.at[0, 'Gebaeudelast']
        Power_to_assign = Power_available
        Power_percent = Power_available / (c * E_EV * 4)
        for i in EV_charging_ID_1:
            df.at[int(index_SOC_1[i]), 'SOC_to_charge_' + str(i)] = Power_percent * SOC_1[i]
        df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

    for i in range(1, Number_EVs + 1):
        if df.at[0, 'Charging_' + str(i)] == 0 and df.at[0, 'SOC_to_charge_' + str(i)] == 0:
            try:
                b = df['SOC_to_charge_' + str(i)].tolist()
                a = next(x for x in b if x > 0)  # 'TODO frage > =
                index_SOC = b.index(a)
                d = df['Charging_' + str(i)].tolist()
                e = next(x for x in d if x > 0)
                index_charging = d.index(e)
                if a * E_EV / (index_SOC - index_charging + 1) > P_max / 4:
                    df.at[int(index_SOC), 'SOC_to_charge_' + str(i)] = P_max / 4 * (
                                index_SOC - index_charging + 1) / E_EV
                    df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)
                c = c + a / (index_SOC + 1)
                len_index_sum = len_index_sum + index_SOC + 1
                index_SOC_1[i] = index_SOC
                SOC_1[i] = a
                SOC_average_1[i] = a / (index_SOC + 1)
            except:
                a = 0
    for i in EV_charging_ID_1:
        b = df['SOC_to_charge_' + str(i)].tolist()
        # locate current SOC to charge
        a = next(x for x in b if x > 0)  # 'TODO frage > =
        index_SOC = b.index(a)
        if a * E_EV / (index_SOC + 1) > P_max / 4:
            df.at[int(index_SOC), 'SOC_to_charge_' + str(i)] = P_max / 4 * (index_SOC + 1) / E_EV
            df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

    EV_charging_ID.clear()
    EV_charging_ID_1.clear()
    SOC_to_charge_sum = 0

    Load_limit.append(df.at[0, 'Grid'] + PV_power_check - int(data_GL.iat[time_check_index_GL, 2]))
    # Load_limit.append(df.at[0, 'Grid'] + real_PV_Globalstrahlung - data_GL.iat[time_check_index_GL, 2])
    load_limit_PV.append(PV_power_check)
    load_limit_GL.append(data_GL.iat[time_check_index_GL, 2])
    load_limit_KED.append(SimConfig.Grid - df.at[0, 'Grid'])
    # Load_limit.append(df.at[0,'Grid']+df.at[0,'PVPower']-df.at[0,'Gebaeudelast']) #TODO ausgabe
    # load_limit_PV.append(df.at[0,'PVPower'])
    # load_limit_GL.append(df.at[0,'Gebaeudelast'])
    # load_limit_KED.append(SimConfig.Grid-df.at[0,'Grid'])
    for i in range(15):
        y_LL_stacked.append(Load_limit[-1])

    # TODO studentierisung
    df = pd.read_csv("./prognose/Prognose_final.csv")
    df['Strompreis'], df['CO2'] = studentisierung(df['Strompreis'], df['CO2'])
    df.to_csv("./prognose/Prognose_final.csv", header=True, index=False)

    # running of the optimization algorithm
    LK = Optimization.Optimierung()

    # storage the timelabel
    xlabel.append(time_label)

    timestamp_check = df1P.at[0, 'TimeIndex']
    # set the timestamp into another form
    time_local = time.localtime(timestamp_check)
    time_check = time.strftime("%d.%m.%Y %H:%M", time_local)
    time_check_index_SP = np.where(data_SP.index == time_check)[0]
    time_check_index_SP = int(time_check_index_SP)
    # if SimConfig.Level == 1 or SimConfig.Forecast_current == 'no':
    #     #Price_Now = 26.807
    #     day = time_local.tm_mday
    #     hour = time_local.tm_hour
    #     minute = time_local.tm_min
    #     row = (day - 1) * 96 + hour * 4 + minute / 15
    #     row = int(row)
    #     df_price = pd.read_csv('./prognose/Eingangsdaten/Strompreis_Prognose_Mittelwert.csv', sep=';', index_col=0)
    #     Price_Now = df_price.iloc[row, 0]
    # else:
    # find the current price in reality
    Price_Now = data_SP.iat[time_check_index_SP, 2]
    # find the energie source in reality
    time_check_index_EQ = np.where(data_EQ.index == time_check)[0]
    time_check_index_EQ = int(time_check_index_EQ)
    # if SimConfig.Level == 1 and SimConfig.Forecast_CO2 == 'no':
    #     day = time_local.tm_mday
    #     hour=time_local.tm_hour
    #     minute=time_local.tm_min
    #     row = (day-1)*96+hour*4+minute/15
    #     row = int(row)
    #     df = pd.read_excel('./prognose/Eingangsdaten/200605_Statistik_Emissionen.xlsx')
    #     CO2_Now = df.iloc[row,2]/3600
    # else:
    #     # #calculate the CO2 Emission in reality
    #     # CO2_Now = (data_EQ.iat[time_check_index_EQ,4]*FaktorBK+data_EQ.iat[time_check_index_EQ\
    #     #                      ,5]*FaktorSK+data_EQ.iat[time_check_index_EQ,6]*FaktorEG+data_EQ.\
    #     #             iat[time_check_index_EQ,7]*FaktorSon)/data_EQ.iat[time_check_index_EQ,8]
    #     CO2_Now = data_EQ.iat[time_check_index_EQ, 3]
    CO2_Now = data_EQ.iat[time_check_index_EQ, 3]

    # if optimazation isnt required:
    if SimConfig.No_Opt == 1:
        # get load limit
        load_limit = Load_limit[-1]
        # count the charging evs, only the first row is required
        if SimConfig.Static_Algo == 'EQUAL':
            list_of_charging_evs_for_equal = []
            for EV in range(1, Number_EVs + 1):
                charging_state = Station[EV - 1][-1]
                # charging_state = df.at[0, 'Charging_%s'%EV]
                if charging_state == 1 and SOC_Now[EV - 1] < 100:
                    list_of_charging_evs_for_equal.append(EV)
            charging_power_equal = LK.equal(len(list_of_charging_evs_for_equal), load_limit)
        else:
            # muss auerßhalb loop zum zählen

            for EV in range(1, Number_EVs + 1):
                charging_state = Station[EV - 1][-1]
                # charging_state = df.at[0, 'Charging_%s' % EV]
                # check if EV is at the station and the SOC is under 100
                if charging_state == 1 and SOC_Now[EV - 1] < 100:
                    # check if the EV is already charging
                    tuple = [item for item in list_charging_evs_with_prio if item[0] == EV]
                    # the EV is already in the last list
                    if tuple:
                        # counting up priority
                        pos = list_charging_evs_with_prio.index(tuple[0])
                        new_prio = tuple[0][1] + 1
                        list_charging_evs_with_prio[pos] = (EV, new_prio)
                    # EV starts charging, add to the list
                    else:
                        list_charging_evs_with_prio.append((EV, 0))
                # EV left/is on a trip
                else:
                    tuple = [item for item in list_charging_evs_with_prio if item[0] == EV]
                    # remove EV from list if it was in the list
                    if tuple:
                        list_charging_evs_with_prio.remove(tuple[0])
            charging_power_fcfs_list = LK.fcfs(load_limit, list_charging_evs_with_prio)
    else:
        ladeplan_opt = LK.ladeplan(Number_EVs)

    for i in range(Number_EVs):
        # storage of outputs of the algorith,
        if SimConfig.No_Opt == 1:
            if SimConfig.Static_Algo == 'EQUAL':
                if i + 1 in list_of_charging_evs_for_equal:
                    power_1st[i] = charging_power_equal
                else:
                    power_1st[i] = 0
            else:
                power_1st[i] = charging_power_fcfs_list[i]

        else:
            power_1st[i] = ladeplan_opt[i]  # LK.ladeplan(Number_EVs)[i]

        # check if SOC will be bigger than 100
        if SOC_Now[i] + power_1st[i] / 4 / E_EV * 100 <= 100:
            # calculation of the charged power
            chargingpower[i] = chargingpower[i] + power_1st[i]
            # buildup of the list of chargingpower
            ChargingPower[i].append(power_1st[i])
            # calculate the current SOC
            SOC_Now[i] = SOC_Now[i] + power_1st[i] / 4 / E_EV * 100  # todo
            # buildup of SOC_List
            SOC_List[i].append(SOC_Now[i])
            # buildup of price list
            Price[i].append(power_1st[i] * Price_Now / 4)
            # buildup of pricesum list
            PriceSum[i].append(Price[i][-1] + PriceSum[i][-1])
            # buildup of CO2_List
            Emission[i].append(power_1st[i] * CO2_Now / 4)
            # buildup of emissionsum list
            EmissionSum[i].append(Emission[i][-1] + EmissionSum[i][-1])
        else:
            # calculate the chargable SOC
            SOC_Diff = 100 - SOC_Now[i]
            # calculate the chargable power
            power_1st[i] = SOC_Diff * E_EV * 0.04
            # calculation of the charged power
            chargingpower[i] = chargingpower[i] + power_1st[i]
            # buildup of the list of chargingpower
            ChargingPower[i].append(power_1st[i])
            # calculate the current SOC
            SOC_Now[i] = 100
            # buildup of SOC_List
            SOC_List[i].append(SOC_Now[i])
            # buildup of price list
            Price[i].append(power_1st[i] * Price_Now / 4)
            # buildup of pricesum list
            PriceSum[i].append(Price[i][-1] + PriceSum[i][-1])
            # buildup of CO2_List
            Emission[i].append(power_1st[i] * CO2_Now / 4)
            # buildup of emissionsum list
            EmissionSum[i].append(Emission[i][-1] + EmissionSum[i][-1])

# letzer ausgerechnter ladeplan ausgeben
if SimConfig.No_Opt == 0:
    last_ladeplan = ladeplan_opt[Number_EVs]


###---------------------------------Bennenung----------------------------------------------------------------------------------------------------
###-----------------------------------------------------------------------------------------------------------------------------------------------
    last_ladeplan.to_csv('./results/'+ str(SimConfig.da) +'_' + str(SimConfig.nu) +'_last_ladeplan.csv',header = True, index = False) # Dateiname
###-------------------------------------------------------------------------------------------------------------------------------------
###-----------------------------------------------------------------------------------------------------------------------------------------------

# storage the last timelabel
timestamp_label = df1P.at[1, 'TimeIndex']
time_label_local = time.localtime(timestamp_label)
time_label = time.strftime("%d.%m.%Y %H:%M", time_label_local)
xlabel.append(time_label)
Load_limit.append(Load_limit[-1])
load_limit_KED.append(load_limit_KED[-1])
load_limit_GL.append(load_limit_GL[-1])
load_limit_PV.append(load_limit_PV[-1])

print(ChargingPower)
# print(TimeCharging)
print(SOC_List)

print(xlabel)
print(PriceSum)

# binding the timelabel, chargingpower and price into prediction result
data = {'Time': xlabel, 'Load_limit': Load_limit, 'PV_Power': load_limit_PV, 'Building_Load': load_limit_GL,
        'KED': load_limit_KED}
df_final = pd.DataFrame(data)
###---------------------------------Bennenung----------------------------------------------------------------------------------------------------
###-----------------------------------------------------------------------------------------------------------------------------------------------
df_final.to_csv('./results/'+ str(SimConfig.da) +'_' + str(SimConfig.nu) + '_' + str(Number_EVs) + 'Grid' + str(SimConfig.Grid) + '.csv', header=True,
                index=False)
###-------------------------------------------------------------------------------------------------------------------------------------
###-----------------------------------------------------------------------------------------------------------------------------------------------
sum_charging_pwr = 0
sum_cost = 0
sum_emission = 0
for i in range(1, Number_EVs + 1):
    for price in Price[i - 1][1:]:
        sum_cost += price
    for emission in Emission[i - 1][1:]:
        sum_emission += emission
    for chrg_pwr in ChargingPower[i - 1]:
        sum_charging_pwr += chrg_pwr
    Charging_Dict = {'Station_' + str(i): Station[i - 1],
                     'ChargingPower_' + str(i): ChargingPower[i - 1],
                     'Price_' + str(i): Price[i - 1][1:], 'CO2_' + str(i): Emission[i - 1][1:],
                     'SOC_' + str(i): SOC_List[i - 1][:-1]}
    df_CP = pd.DataFrame(Charging_Dict)
    df_CP.to_csv('Result_charging.csv', header=True, index=False)
    col_name1 = df_final.columns.tolist()
    col_name2_1 = str('ChargingPower_') + str(i)
    col_name2_2 = str('Price_') + str(i)

    col_name1.append(str('Station_') + str(i))
    col_name1.append(col_name2_1)
    col_name1.append(col_name2_2)

    df_final = df_final.reindex(columns=col_name1)
    df_final[str('Station_') + str(i)] = df_CP[str('Station_') + str(i)]
    df_final[str('ChargingPower_') + str(i)] = df_CP[str('ChargingPower_') + str(i)]
    df_final[str('Price_') + str(i)] = df_CP[str('Price_') + str(i)]
    df_final[str('CO2_') + str(i)] = df_CP[str('CO2_') + str(i)]
    df_final[str('SoC_') + str(i)] = df_CP[str('SOC_') + str(i)]
    ###---------------------------------Bennenung----------------------------------------------------------------------------------------------------
    ###-----------------------------------------------------------------------------------------------------------------------------------------------
    df_final.to_csv('./results/'+ str(SimConfig.da) +'_' + str(SimConfig.nu) + '_' + str(SimConfig.Level) + '_EV' +
                    ###-------------------------------------------------------------------------------------------------------------------------------------
                    ###-----------------------------------------------------------------------------------------------------------------------------------------------
                    str(Number_EVs) + 'Grid' + str(SimConfig.Grid) + str(OTM_Target) +
                    'C' + str(SimConfig.Forecast_current) +
                    'B' + str(SimConfig.Forecast_building_load) +
                    'E' + str(SimConfig.Forecast_CO2) +
                    'P' + str(SimConfig.Forecast_PV) +
                    '.csv', header=True, index=False)
df_final['sum_charging']=''
df_final['sum_charging'][0] = sum_charging_pwr

df_final['sum_cost']=''
df_final['sum_cost'][0] = sum_cost

df_final['sum_co2']=''
df_final['sum_co2'][0] = sum_emission
###---------------------------------Bennenung----------------------------------------------------------------------------------------------------
####-----------------------------------------------------------------------------------------------------------------------------------------------
df_final.to_csv('./results/'+ str(SimConfig.da) +'_' + str(SimConfig.nu) +'_Result_final.csv' ,header = True, index = False)
# set the labels
labels = []
xlabel_stacked = []

# reconstructure SOC


# set the labels for each vehicle
for i in range(1, Number_EVs + 1):
    labels.append('EV' + str(i))

# stack the charging power for every minute for stackplot
for i in range(Number_EVs):
    for j in range(1, int(96 * SimDuration)):
        for k in range(15):
            y_CP_stacked[i].append(ChargingPower[i][j])

# define the xlabel for charging power after stacking
xlabel_stacked = range(int(SimDuration * 24 * 60))

# assign the time now
now = time.strftime("%d-%m-%Y-%H_%M_%S", time.localtime(time.time()))
# setup subplot parameter
pars = SubplotParams(left=0.09, right=0.8)
# define color pattern
pal = ["#005374", "#B00046", "#007156", "#760054", "#ffcd00", "#C6EE00", "#969696",
       "#FA6E00", "#CC0099", "#C4D17F", "#7CCDE6", "#89A400", "#BE1E3C", "#8CB1C0",
       "#BA7FBA", "#FDD3B2", "#ADBF4D", "#D6B2D6", "#DDDDDD", "#FFE67F"]


# define random colors if pal is not enough
def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


if Number_EVs > 20:
    color_extra = randomcolor()
    pal.append(str(color_extra))
# plotting charging power
fig, ax = plt.subplots(subplotpars=pars, figsize=(9, 5), dpi=100, facecolor='w',
                       edgecolor='k')
ax.stackplot(xlabel_stacked, y_CP_stacked, labels=labels, colors=pal,
             baseline='zero', alpha=0.6)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5, prop={'size': 9})
ax.plot(xlabel_stacked, y_LL_stacked)
ax.text(20, y_LL_stacked[20] + 0.5, 'Ladelimit', fontsize=10, weight='bold')
xticks = list(range(0, int(SimDuration * 96 * 15), int(SimDuration * 24 * 15)))
xlabels = [xlabel[int(x / 15)] for x in xticks]
xticks.append(SimDuration * 96 * 15)
xlabels.append(xlabel[-1])
plt.xlabel('Zeit', horizontalalignment='right')
plt.ylabel('Ladeleistung in kW')
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(40))
plt.title("Summierte Ladeleistung", fontsize=10, fontweight='bold', pad=10)
plt.savefig('./fig/ChargingPower_' + str(now) + '.png')
plt.show()

# plotting current cost
fig, ax = plt.subplots(subplotpars=pars, figsize=(9, 5), dpi=100, facecolor='w',
                       edgecolor='k')
ax.stackplot(xlabel, PriceSum, labels=labels, colors=pal,
             baseline='zero', alpha=0.6)
ax.legend(loc='upper left')
xticks = list(range(0, int(SimDuration * 96), int(SimDuration * 24)))
xlabels = [xlabel[int(x)] for x in xticks]
xticks.append(SimDuration * 96)
xlabels.append(xlabel[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
plt.xlabel('Zeit', horizontalalignment='right')
plt.ylabel('Ladeleistung in Cent')
plt.title("Summierte Ladekosten", fontsize=10, fontweight='bold', pad=10)
plt.savefig('./fig/CurrentCost_' + str(now) + '.png')
plt.show()

# plotting CO2
fig, ax = plt.subplots(subplotpars=pars, figsize=(9, 5), dpi=100, facecolor='w',
                       edgecolor='k')
ax.stackplot(xlabel, EmissionSum, labels=labels, colors=pal,
             baseline='zero', alpha=0.6)
ax.legend(loc='upper left')
xticks = list(range(0, int(SimDuration * 96), int(SimDuration * 24)))
xlabels = [xlabel[int(x)] for x in xticks]
xticks.append(SimDuration * 96)
xlabels.append(xlabel[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
plt.xlabel('Zeit', horizontalalignment='right')
plt.ylabel('CO2-Emission in g')
plt.title("Summierte CO2-Emissionen", fontsize=10, fontweight='bold', pad=10)
plt.savefig('./fig/CO2-Emission_' + str(now) + '.png')
plt.show()

# plotting SOC of each EV
fig1, ax = plt.subplots(subplotpars=pars, figsize=(9, 5), dpi=100, facecolor='w',
                        edgecolor='k')
xticks = list(range(0, int(SimDuration * 96), int(SimDuration * 24)))
xlabels = [xlabel[int(x)] for x in xticks]
xticks.append(SimDuration * 96)
xlabels.append(xlabel[-1])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
for i in range(Number_EVs):
    ax.plot(xlabel, SOC_List[i], label=labels[i], color=pal[i], linewidth=2, alpha=0.8)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5, prop={'size': 9})
ax.set_ylabel('Ladezustand (SoC) in %')
ax.set_title('Ladezustand (SoC)', fontsize=10, fontweight='bold', pad=10)
plt.savefig('./fig/SOC_' + str(now) + '.png')
plt.show()
