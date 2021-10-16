import pandas as pd
import os

path = 'Diagramm/' # Daten von CO2; wenn Strompreis, bitte auf 'Daten/SP/' ändern
files = os.listdir(path)

# Start_Datum eingeben:
Start_day = 15
Start_month = 4
Start_year = 2019

# End_Datum eingeben:
#Enddatum ist inklusive
End_day = 20
End_month = 4
End_year = 2019

all_values = []

for year in range(Start_year, End_year + 1, 1):
    for month in range(Start_month, End_month + 1, 1):
        for day in range(Start_day, End_day + 1, 1):
            if len(str(month)) == 1:
                month_string = '0' + str(month)
            else:
                month_string = str(month)
            if len(str(day)) == 1:
                day_string = '0' + str(day)
            else:
                day_string = str(day)
            Date = 'SP_' + day_string + '_' + month_string + '_' + str(year) # wenn Strompreis, bitte 'CO2_' auf 'SP_'
            for filename in files:
                if filename.startswith(Date):
                    df = pd.read_csv(path + filename)
                    all_values += df.values.tolist()
maximum = max(all_values)
minimum = min(all_values)

print('Maximum:' + str(maximum))
print('Minimum:' + str(minimum))

