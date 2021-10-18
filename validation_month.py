import prognose.PrognoseSP_New as PrognoseSP
import prognose.PrognoseGL_New as PrognoseGL
import prognose.PrognosePV_test as PrognosePV
import prognose.PrognoseCO2 as PrognoseCO2

import datetime
import pandas as pd
import numpy as np


def main(modul):
    dates = pd.date_range(start="2019-04-02 0:00",end="2019-04-30 0:00", freq='D')
    #print(dates)
    values = []
    for date in dates:
        timeindex = str(date.strftime('%d.%m.%Y %H:%M'))
        print(timeindex)
        #res = PrognoseGL.forecast(timeindex)
       # res = PrognosePV.forecast(timeindex)
        res =  PrognoseCO2.forecast(timeindex)
        #res = modul.run_forecast(timeindex)
        #res = modul.run(timeindex)
        values.append(res)

    print(values)
    print('mean ', np.mean(values))




if __name__ == '__main__':
    #prognose_test = PrognosePV.PrognosePV()
    #prognose_test = PrognoseSP.PrognoseSP( )
    prognose_test = PrognoseGL.PrognoseGL()
   # prognose_test = PrognoseCO2.
    main(prognose_test)