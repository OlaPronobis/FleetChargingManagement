import pandas as pd
import time
from datetime import datetime
import datetime
import random
import matplotlib.pyplot as plt


# ein String eines Zeitpunktes wie 'hh:mm' in die Werte
def time_to_value(timestring):
    timeinfo = timestring.split(':')
    hours, minutes = int(timeinfo[0]), int(timeinfo[1])
    if hours > 24 or minutes >= 60:
        return None
    else:
        value = hours / 24 + (minutes / 60 / 24)
        return value


# einen Wert zw. 0 u. 1 in die Format der Uhrzeit hh:mm
def value_to_time(timevalue):
    if timevalue >= 1:
        if timevalue >= 2:
            timevalue -= 1
        new_timevalue = timevalue - 1
        hours = int(new_timevalue * 24)
        minutes = int((new_timevalue * 24 - hours) * 60)
        timeinfo = str(hours) + ':' + str(minutes)
        timestring = datetime.datetime.strptime(timeinfo, '%H:%M')
        return timestring.strftime('%H:%M')
    elif timevalue >= 0:
        hours = int(timevalue * 24)
        minutes = int((timevalue * 24 - hours) * 60)
        timeinfo = str(hours) + ':' + str(minutes)
        timestring = datetime.datetime.strptime(timeinfo, '%H:%M')
        return timestring.strftime('%H:%M')
    else:
        return None


def generator(low, upper, number, digits):
    cache = []
    for x in range(number):
        random.seed()  # Initialisierung RNG
        cache.append(round(random.uniform(low, upper), digits))
    return cache


def verteilung(data, bins, amount, digits):
    minimum = min(data)
    maximum = max(data)
    number_of_data = len(data)
    interval = []
    data.sort()
    for i in range(bins):
        interval.append(minimum + i * (maximum - minimum) / bins)
    interval.append(maximum)
    results_pool = []
    numbers = 0
    for n in range(1, bins + 1):
        for i in range(number_of_data):
            if interval[n] >= data[i] >= interval[n - 1]:
                numbers += 1
            else:
                results_pool += generator(interval[n - 1], interval[n], round(amount * numbers / number_of_data),
                                          digits)
                numbers = 0
    return results_pool

### Generator###
df = pd.read_excel('KED.xlsx')
df_trips = pd.read_excel('Trips_KED.xlsx')
df['Abfahrt'] = pd.to_datetime(df['Abfahrt'])
df['Ankunft'] = pd.to_datetime(df['Ankunft'])
df['Abfahrt_Values'] = df['Abfahrt'].dt.hour / 24 + df['Abfahrt'].dt.minute / 60 / 24
df['Ankunft_Values'] = df['Ankunft'].dt.hour / 24 + df['Ankunft'].dt.minute / 60 / 24
df['Standzeit'] = df['Standzeit'] / 24
df['Fahrzeit'] = df['Ankunft_Values'] - df['Abfahrt_Values']
df['Diff_km'] = - df['Diff_km']
df['Diff_SoC'] = 31.08*df['Diff_km']/94
df = df[~(df['Fahrzeit'] < 0)]
df = df[~(df['Diff_SoC'] < 0)]
df = df[~(df['Diff_km'] < 0)]

list_ZP_Abfahrt = df['Abfahrt_Values'].values.tolist()
verteilung_Abfahrt = verteilung(list_ZP_Abfahrt, 24, 20000, 6)
list_ZP_Ankunft = df['Ankunft_Values'].values.tolist()
#verteilung_Ankunft = verteilung(list_ZP_Ankunft, 24, 2000, 7)
list_trips = df_trips['total_Trips_KED'].values.tolist()
verteilung_trips = verteilung(list_trips, 16, 5000, 1)
list_Standzeit = df['Standzeit'].values.tolist()
verteilung_Standzeit = verteilung(list_Standzeit, 24, 20000, 6)
list_Fahrzeit = df['Fahrzeit'].values.tolist()
verteilung_Fahrzeit = verteilung(list_Fahrzeit, 24, 20000, 6)
list_Start_SoC = df['Start_SoC'].values.tolist()
verteilung_Start_SoC = verteilung(list_Start_SoC, 100, 2000, 2)
list_Diff_SoC = df['Diff_SoC'].values.tolist()
verteilung_Diff_SoC = verteilung(list_Diff_SoC, 100, 4000, 6)
'''
fig1, (ax1, ax2) = plt.subplots(2, 1)
n1, bins1, patches1 = ax1.hist(list_Diff_SoC, rwidth=0.85, bins=20)
ax1.set_title('Verteilung der SoC-Differenz')
ax1.set_xlabel('')
n2, bins2, patches2 = ax2.hist(verteilung_Diff_SoC, rwidth=0.85, bins=20)
ax2.set_title('Verteilung der SoC-Differenz (generiert)')
ax2.set_xlabel('')

fig2, (ax3, ax4) = plt.subplots(2, 1)
n3, bins3, patches3 = ax3.hist(list_trips, rwidth=0.85, bins=16)
ax3.set_title('Verteilung der Trips pro Tag')
ax3.set_xlabel('Anzahl Trips')
n4, bins4, patches4 = ax4.hist(verteilung_trips, rwidth=0.85, bins=16)
ax4.set_title('Verteilung der Trips pro Tag (generiert)')
ax4.set_xlabel('Anzahl Trips')

fig3, (ax5, ax6) = plt.subplots(2, 1)
n5, bins5, patches5 = ax5.hist(list_Standzeit, rwidth=0.85, bins=20, range=(0, 1))
ax5.set_title('Verteilung der Standzeiten')
ax5.set_xlabel('Standzeiten in 1/24-Wert')
n6, bins6, patches6 = ax6.hist(verteilung_Standzeit, rwidth=0.85, bins=20, range=(0, 1))
ax6.set_title('Verteilung der Standzeiten (generiert)')
ax6.set_xlabel('Standzeiten in 1/24-Wert')

plt.show()
'''
###### Eingabe ######

#Anfangszeit_von = '8:00'  # Anfangszeit hier eingeben, in Format "hh:mm"
#Anfangszeit_bis = '12:00'  # Anfangszeit hier eingeben, in Format "hh:mm"


Anfangsdatum = datetime.datetime.strptime('31-12-2018', '%d-%m-%Y')
Anzahl_Fahrzeuge = 48  # Anzahl Fahzeuge hier eingeben
Anzahl_Tage = 370      # Anzahl Tage



#Anfang_von = time_to_value(Anfangszeit_von)
#Anfang_bis = time_to_value(Anfangszeit_bis)

now_time = time.strftime("%d-%m-%Y %H_%M_%S")

for EV in range(Anzahl_Fahrzeuge):
    df_results = pd.DataFrame(columns=['Ankunft','LadeStart', 'Abfahrt','LadeEnde', 'Ankunft_Values', 'Abfahrt_Values', 'Fahrzeit', 'Ladezeit', 'Start_SoC', 'End_SoC', 'Verbrauch_SoC'])
    Zeilen = 0
    for simulationstag in range(Anzahl_Tage):
        day = 0
        Datum = Anfangsdatum + datetime.timedelta(days=simulationstag)
        Trips = round(random.choice(verteilung_trips))
        if Trips == 0:
            Trips += 1
        Anfang = random.sample(verteilung_Abfahrt, k=Trips)
        Anfang.sort()

        for trips in range(Trips):
            df_results.loc[Zeilen + trips, 'Abfahrt_Values'] = Anfang[trips]
            df_results.loc[Zeilen + trips, 'Abfahrt'] = value_to_time(Anfang[trips])
            df_results.loc[Zeilen + trips, 'Abfahrt'] = Datum.strftime('%d-%m-%Y') + ' ' + value_to_time(Anfang[trips])
            timeArray = time.strptime(df_results.loc[Zeilen + trips, 'Abfahrt'], "%d-%m-%Y %H:%M")
            timestamp = time.mktime(timeArray)
            df_results.loc[Zeilen + trips, 'LadeEnde'] = timestamp
            df_results.loc[Zeilen + trips, 'Start_SoC'] = random.choice(verteilung_Start_SoC)
            if trips == Trips-1:
                df_results.loc[Zeilen + trips, 'Fahrzeit'] = random.choice(verteilung_Fahrzeit)
            else:
                df_results.loc[Zeilen + trips, 'Fahrzeit'] = random.choice(
                    [item for item in verteilung_Fahrzeit if item < Anfang[trips + 1] - Anfang[trips]])

            if df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:05'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice([item for item in verteilung_Diff_SoC if item < 1.37765957])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:10'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice([item for item in verteilung_Diff_SoC if item < 2.75531915])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:15'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 4.13297872])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:20'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 5.5106383])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:25'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 6.88829787])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:30'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 8.26595745])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:35'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 9.64361702])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:40'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 11.0212766])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:45'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 12.3989362])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:50'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 13.7765957])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('00:55'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 15.1542553])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('01:00'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 16.5319149])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('01:30'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 24.7978724])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('02:00'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 33.0638298])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('02:30'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 41.3297872])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('03:00'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 49.5957447])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('03:30'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 24.7978724])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('04:00'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 3.0638298])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('04:30'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 41.3297872])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('05:00'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 82.6595745])
            elif df_results.loc[Zeilen + trips, 'Fahrzeit'] <= time_to_value('05:30'):
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(
                    [item for item in verteilung_Diff_SoC if item < 90.925532])
            else:
                df_results.loc[Zeilen + trips, 'Verbrauch_SoC'] = random.choice(verteilung_Diff_SoC)
            df_results.loc[Zeilen + trips, 'End_SoC'] = df_results.loc[Zeilen + trips, 'Start_SoC'] - df_results.loc[
                Zeilen + trips, 'Verbrauch_SoC']

            if Zeilen == 0:
                df_results.loc[Zeilen, 'Ladezeit'] = random.choice(verteilung_Standzeit)
                df_results.loc[Zeilen, 'Ankunft_Values'] = df_results.loc[Zeilen, 'Abfahrt_Values'] - df_results.loc[Zeilen, 'Ladezeit']
                if df_results.loc[Zeilen, 'Ankunft_Values'] < 0:
                    df_results.loc[Zeilen, 'Ankunft_Values'] = 0
                df_results.loc[Zeilen, 'Ankunft'] = Datum.strftime('%d-%m-%Y') + ' ' + value_to_time(df_results.loc[Zeilen, 'Ankunft_Values'])
                timeArray = time.strptime(df_results.loc[Zeilen, 'Ankunft'], "%d-%m-%Y %H:%M")
                timestamp = time.mktime(timeArray)
                df_results.loc[Zeilen, 'LadeStart'] = timestamp

            df_results.loc[Zeilen + trips + 1, 'Ankunft_Values'] = df_results.loc[Zeilen + trips, 'Abfahrt_Values'] + df_results.loc[Zeilen + trips, 'Fahrzeit']

            df_results.loc[Zeilen + trips + 1, 'Ankunft'] = Datum.strftime('%d-%m-%Y') + ' ' + value_to_time(
                    df_results.loc[Zeilen + trips + 1, 'Ankunft_Values'])
            timeArray = time.strptime(df_results.loc[Zeilen + trips + 1, 'Ankunft'], "%d-%m-%Y %H:%M")
            timestamp = time.mktime(timeArray)
            df_results.loc[Zeilen + trips + 1, 'LadeStart'] = timestamp

            df_results.loc[Zeilen + trips, 'Ladezeit'] = df_results.loc[Zeilen + trips, 'Abfahrt_Values'] - df_results.loc[Zeilen + trips, 'Ankunft_Values']
            if df_results.loc[Zeilen + trips, 'Ladezeit'] < 0:
                df_results.loc[Zeilen + trips, 'Ladezeit'] = df_results.loc[Zeilen + trips, 'Ladezeit'] + 1
            ####### O Uhr zu 0 Uhr Tag ##########
            if df_results.loc[Zeilen + trips, 'Ankunft_Values'] > time_to_value('24:00'):
                df_results.loc[Zeilen + trips, 'Ankunft_Values'] = 1
                df_results.loc[Zeilen + trips, 'Ankunft'] = Datum.strftime('%d-%m-%Y') + ' ' + value_to_time(1)
                timeArray = time.strptime(df_results.loc[Zeilen + trips, 'Ankunft'], "%d-%m-%Y %H:%M")
                timestamp = time.mktime(timeArray)
                df_results.loc[Zeilen, 'LadeStart'] = timestamp

                df_results.loc[Zeilen + trips, 'Fahrzeit'] = 1 - df_results.loc[Zeilen + trips, 'Abfahrt_Values']

        Zeilen = len(df_results)-1

    df_results['Fahrzeit'] = df_results['Fahrzeit'] * 96 # zu 15min Wert
    df_results['Ladezeit'] = df_results['Ladezeit'] * 96
    #df_results['Abfahrt_Values'] = df_results['Abfahrt_Values'] * 96
    #df_results['Ankunft_Values'] = df_results['Ankunft_Values'] * 96
    df_results.drop(df_results.index[len(df_results) - 1])
    df_results.to_csv('TestScenario/KED/'+ 'TestScenario' + str(EV+1) + '.csv', index=0)
    df_results.drop(df_results.index, inplace=True)


# funktion to generate the final ked csv saved to TestScenario\KED\KED_final.csv
# will be imported to the main_simulation
# a summarized version of the generated data for KED EVs, columns: 'Date' 'Timestamp' 'Charge_EV_x' 'SOC_Usage_EV_x'
# if ev is charging: Charge -> 1 else 0
# in the first row with charging == 0 the SOC usage of the next trip is saved
def generate_final_ked_csv():
    start_date = Anfangsdatum
    days = Anzahl_Tage
    duration = days * 96
    date_range = pd.date_range(start=start_date, periods=duration, freq='15 min')
    full_dates = []
    for dates in date_range:
        full_dates.append(datetime.datetime.timestamp(dates))
    df_final = {'Date': date_range, 'Timestamp': full_dates}
    df_final = pd.DataFrame(df_final)
    for x in range(1,Anzahl_Fahrzeuge+1):
        df = pd.read_csv("./TestScenario/KED/TestScenario"+ str(x)+".csv")
        df_KED = []
        SOC_Verbrauch_cache = []
        # iterate over the timestamp (15 min interval)
        for ts in full_dates:
            KED_in_ts = False
            # search for stops at charging station that are around the timestamp
            for row in range(len(df)):
                ladestart = df.loc[row, 'LadeStart']
                ladeende = df.loc[row, 'LadeEnde']
                # ev charging start must be after timestamp and charging stop must be after the next timestamp
                if ladestart <= ts and ladeende >= (ts+15*60):
                    KED_in_ts = True
                    # save the SOC usage for later
                    next_SOC_Verbrauch = df.loc[row, 'Verbrauch_SoC']
                    break
            if KED_in_ts:
                # if KED is at charging station in timestamp
                df_KED.append(1)
                SOC_Verbrauch_cache.append(next_SOC_Verbrauch)
            else:
                # if KED already left the charging station in timestamp
                df_KED.append(0)
                SOC_Verbrauch_cache.append('')
                #else: ### was wenn innerhalb der 15 min weg?
        # set the SOC usage into the row where the ev first left after visiting the charging station
        SOC_Verbrauch = ['']
        for row in range(1,len(df_KED)):
            # find the first row after ev left (first 0 after 1)
            if df_KED[row] == 0 and df_KED[row-1] ==1:
                # set the value to the soc_usage of the last row
                SOC_Verbrauch.append(SOC_Verbrauch_cache[row-1])
            else:
                SOC_Verbrauch.append('')
        # save the final csv
        df_final['Charge_EV_%s'%x] = df_KED
        df_final['SOC_Usage_EV_%s' % x] = SOC_Verbrauch
        df_final.to_csv('./TestScenario/KED/KED_final.csv', index=False)

generate_final_ked_csv()

