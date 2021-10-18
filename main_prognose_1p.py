import pandas as pd
import numpy as np
import datetime
import time
# import prognose.PrognoseSP as PrognoseSP
# import prognose.PrognoseGL as PrognoseGL
# import prognose.PrognoseSonstig as PrognoseSonstig
# import prognose.PrognoseBK as PrognoseBK
# import prognose.PrognoseSK as PrognoseSK
# import prognose.PrognoseEG as PrognoseEG
# import prognose.PrognoseGE as PrognoseGE
# #import prognose.PrognosePV as PrognosePV
# import prognose.GestaltungFahrplan as GestaltungFahrplan
import SimulationConfiguration as SimConfig

import prognose.PrognoseSP_New as PrognoseSP
import prognose.PrognoseGL_New as PrognoseGL
import prognose.PrognosePV_New as PrognosePV
import prognose.PrognoseCO2 as PrognoseCO2


class Prognosis1P():
    def __init__(self):
        self.PV = PrognosePV.PrognosePV()  # hat das if schon in run
        self.SP = PrognoseSP.PrognoseSP()
        self.GL = PrognoseGL.PrognoseGL()  # hat das if schon in run
        self.CO2 = PrognoseCO2.PrognoseCO2()

        # self.FaktorBK = 407
        # self.FaktorSK = 336
        # self.FaktorEG = 201
        # self.FaktorSon = 288

    # forecast_list: a list of the objects that needs a new forecast model, if object is not in the list
    # then it will use the current model and compute new values
    def run(self, timeindex, localtime, forecast_list=None):
        # GF = GestaltungFahrplan.GestaltungFahrplan()
        # GF.Gestaltung()

        grid = []
        Grid = SimConfig.Grid

        for n in range(SimConfig.Forecast_out):
            grid.append(Grid)

        begin = datetime.datetime(localtime.tm_year, localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                  localtime.tm_min)
        end = begin + datetime.timedelta(days=SimConfig.Forecast_out / 96)
        timeindex_final = []
        d = begin
        delta = datetime.timedelta(minutes=15)
        while d < end:
            t = d.timetuple()
            timeStamp = int(time.mktime(t))
            timeStamp = float(str(timeStamp) + str("%06d" % d.microsecond)) / 1000000
            timeindex_final.append(timeStamp)
            # timeindex.append(d.strftime("%Y-%m-%d %H:%M"))
            d += delta

        # PV = PrognosePV.PrognosePV()
        if 'PV' in forecast_list:
            self.PV.run(timeindex)
        else:
            self.PV.run(timeindex, new_svr=False)
        PVPower = pd.read_csv('./prognose/Prognose_PVPower.csv', usecols=[0])
        PVPower = np.array(PVPower)
        PVPower = PVPower.flatten()
        PVPower = PVPower.tolist()
        if SimConfig.Forecast_current == 'no' or SimConfig.Level == 1:
            # Strompreis = [26.807]*96
            timeArray = time.strptime(timeindex, "%d.%m.%Y %H:%M")
            timestamp = time.mktime(timeArray)
            time_local = time.localtime(timestamp)
            day = time_local.tm_mday
            hour = time_local.tm_hour
            minute = time_local.tm_min
            row = (day - 1) * 96 + hour * 4 + minute / 15
            row = int(row)
            Strompreis = []
            df = pd.read_csv('./prognose/Eingangsdaten/Strompreis_Prognose_Mittelwert.csv', sep=';')
            for i in range(row, row + SimConfig.Forecast_out):
                Strompreis.append(df.iloc[i, 1])
        else:
            # SP = PrognoseSP.PrognoseSP()
            if 'SP' in forecast_list or self.SP.svr_rbf is None:
                self.SP.run(timeindex)
            else:
                self.SP.get_new_values(timeindex)
            Strompreis = pd.read_csv('./prognose/Prognose_Strompreis.csv', usecols=[0])
            Strompreis = np.array(Strompreis)
            Strompreis = Strompreis.flatten()
            Strompreis = Strompreis.tolist()

        # GL = PrognoseGL.PrognoseGL()
        if 'GL' in forecast_list:
            self.GL.run(timeindex)
        else:
            self.GL.run(timeindex, new_svr=False)
        Gebaeudelast = pd.read_csv('./prognose/Prognose_Gebaeudelast.csv', usecols=[0])
        Gebaeudelast = np.array(Gebaeudelast)
        Gebaeudelast = Gebaeudelast.flatten()
        Gebaeudelast = Gebaeudelast.tolist()

        if SimConfig.Forecast_CO2 == 'no' or SimConfig.Level == 1:
            timeArray = time.strptime(timeindex, "%d.%m.%Y %H:%M")
            timestamp = time.mktime(timeArray)
            time_local = time.localtime(timestamp)
            day = time_local.tm_mday
            hour = time_local.tm_hour
            minute = time_local.tm_min
            row = (day - 1) * 96 + hour * 4 + minute / 15
            row = int(row)
            CO2 = []
            df = pd.read_excel('./prognose/Eingangsdaten/200605_Statistik_Emissionen.xlsx')
            for i in range(row, row + SimConfig.Forecast_out):
                CO2.append(df.iloc[i, 1])
        else:
            if 'CO2' in forecast_list or self.CO2.svr_rbf is None:
                self.CO2.run(timeindex)
            else:
                self.CO2.get_new_values(timeindex)
            CO2 = pd.read_csv('./prognose/Prognose_CO2.csv', usecols=[0])
            CO2 = np.array(CO2)
            CO2 = CO2.flatten()
            CO2 = CO2.tolist()

            # BK = PrognoseBK.PrognoseBK()
            # BK.run(timeindex)
            # Braunkohle = pd.read_csv('./prognose/Prognose_Braunkohle.csv',usecols=[0])
            # Braunkohle = np.array(Braunkohle)
            #
            # SK = PrognoseSK.PrognoseSK()
            # SK.run(timeindex)
            # Steinkohle = pd.read_csv('./prognose/Prognose_Steinkohle.csv',usecols=[0])
            # Steinkohle = np.array(Braunkohle)
            #
            # EG = PrognoseEG.PrognoseEG()
            # EG.run(timeindex)
            # Erdgas = pd.read_csv('./prognose/Prognose_Erdgas.csv',usecols=[0])
            # Erdgas = np.array(Erdgas)
            #
            # Son = PrognoseSonstig.PrognoseSonstig()
            # Son.run(timeindex)
            # SonstigeKonventionelle = pd.read_csv('./prognose/Prognose_SonstigeKonventionelle.csv',usecols=[0])
            # SonstigeKonventionelle = np.array(SonstigeKonventionelle)
            #
            # GE = PrognoseGE.PrognoseGE()
            # GE.run(timeindex)
            # GesamteerzeugteEnergie = pd.read_csv('./prognose/Prognose_GesamteerzeugteEnergie.csv',usecols=[0])
            # GesamteerzeugteEnergie = np.array(Erdgas)

            # CO2 = (self.FaktorBK*Braunkohle+self.FaktorSK*Steinkohle+self.FaktorEG*Erdgas\
            #       +self.FaktorSon*SonstigeKonventionelle)/GesamteerzeugteEnergie #g/kWh
            # CO2 = CO2.flatten()
            # CO2 = CO2.tolist()

        data = {'TimeIndex': timeindex_final, 'Grid': grid, 'PVPower': PVPower, 'Gebaeudelast': Gebaeudelast,
                'Strompreis': Strompreis, 'CO2': CO2}
        # print(len(timeindex_final))
        # print(len(grid))
        # print(len(PVPower))
        # print(len(Gebaeudelast))
        # print(len(Strompreis))
        # print(len(CO2))
        # print(data)
        df = pd.DataFrame(data)

        df.to_csv("Prognose_1P.csv", header=True, index=False)

        '''for x in range(nb_of_EVs):
            col_name1 = df.columns.tolist()                   

            col_name2_1 = str('Charging_')+str(int(x))
            col_name2_2 = str('SOC_to_charge_')+str(int(x))

            col_name1.append(col_name2_1)
            col_name1.append(col_name2_2)

            df=df.reindex(columns=col_name1)

            FPC = pd.read_csv('./prognose/Prognose_EV.csv',usecols=[int(2*x)])
            FPC = np.array(FPC)
            FPC = FPC.flatten()
            FPC = FPC.tolist()
            FPS = pd.read_csv('./prognose/Prognose_EV.csv',usecols=[int(2*x+1)])
            FPS = np.array(FPS)
            FPS = FPS.flatten()
            FPS = FPS.tolist()

            df[str('Charging_')+str(int(x))]=FPC
            df[str('SOC_to_charge_')+str(int(x))]=FPS
            df.to_csv("Prognose_test.csv",header = True, index = False)'''