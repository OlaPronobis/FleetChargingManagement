import pandas as pd
import datetime
df = pd.read_csv("./prognose/Prognose_final.csv")
EV_charging_ID = [1,2,3,4]
E_EV = 94
P_max = 11
c = 0
len_index_sum = 0
index_SOC_1 = {}
SOC_1 = {}
SOC_average_1 = {}
for i in EV_charging_ID:
    b = df['SOC_to_charge_'+str(i)].tolist()
    #locate current SOC to charge
    a = next(x for x in b if x > 0)
    index_SOC = b.index(a)
    if a*E_EV/(index_SOC+1)>P_max/4:
        df.at[int(index_SOC),'SOC_to_charge_'+str(i)]=P_max/4*(index_SOC+1)/E_EV
        df.to_csv("./prognose/Prognose_final.csv",header = True, index = False)
    c = c + a/(index_SOC+1)
    len_index_sum = len_index_sum + index_SOC+1
    index_SOC_1[i] = index_SOC
    SOC_1[i] = a
    SOC_average_1[i] = a/(index_SOC+1)

if c*E_EV*4>df.at[0,'Grid'] + df.at[0,'PVPower'] - df.at[0,'Gebaeudelast']:
    Power_available = df.at[0,'Grid'] + df.at[0,'PVPower'] - df.at[0,'Gebaeudelast']
    Power_to_assign = Power_available
    Power_percent = Power_available/(c*E_EV*4)
    for i in EV_charging_ID:
        df.at[int(index_SOC),'SOC_to_charge_'+str(i)]=Power_percent*a
    df.to_csv("./prognose/Prognose_final.csv",header = True, index = False)