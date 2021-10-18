import datetime

import SimulationConfiguration as SimConfig

class TimeIndex():
    
    def __int__(self,time=SimConfig.time):
        self.begin = datetime.datetime(time.year,time.month,time.day,time.hour,time.minute)
        self.end = self.begin+datetime.timedelta(days=1)
    
    def run(self):
        timeindex = []
        d = self.begin
        delta = datetime.timedelta(minutes=15)
        while self.d < self.end:
            timeindex.append(self.d.strftime("%Y-%m-%d %H:%M"))
            d += delta
        return timeindex