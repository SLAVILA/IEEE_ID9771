import calendar
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

# +
class ClimatologiaLoc():
    def __init__(self, clim):
        self.clim = clim
    
    def __getitem__(self, key):
        return self.clim[key]

class Climatologia():
    def __init__(self, df_orig):
        self.leap = 2000
        self.normal = 1999
        self.years = {
            "leap": 2000,
            "normal": 1999
        }
        self.tiles = {}
        for name, year in (("leap", self.years["leap"]), ("normal", self.years["normal"])):
            df = df_orig.rename(columns=lambda x: datetime.datetime(year, int(x), 15-(x=="2")))
            df = df.reindex(sorted(df.columns), axis=1)
            self.tiles[name] = self.__interpolate_helper(df)
            
    @staticmethod
    def __interpolate_helper(df):
        # Adiciona pontos antes e dps 
        for d, dy in [(df.columns[-i], -1) for i in range(1, 4)] + [(df.columns[i], +1) for i in range(3)]:
            df[datetime.datetime(d.year+dy, d.month, d.day)] = df[d]
        df = df.reindex(sorted(df.columns), axis=1).T
        df = df.resample("D").asfreq().interpolate(method='polynomial', order=2)
        return df.loc[datetime.datetime(df.index[0].year+1, 1, 1): datetime.datetime(df.index[0].year+1, 12, 31)]

    @property
    def loc(self):
        return ClimatologiaLoc(self)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return self(start, stop).loc[key]
    
    def __call__(self, d_inicial, d_final):
        if (d_inicial.year == d_final.year):
            year_type = "leap" if calendar.isleap(d_inicial.year) else "normal"
            year = self.years[year_type]
            
            date = pd.date_range(
                start=d_inicial.replace(year=year), 
                end=d_final.replace(year=year))
            temp = self.tiles[year_type].loc[date]
            
            delta_time = \
                datetime.datetime(year=d_inicial.year, month=1, day=1) -\
                datetime.datetime(year=year, month=1, day=1)
            temp.index += delta_time
            return temp
            
        years = [
            self(d_inicial, d_inicial.replace(month=12, day=31)),
            self(d_final.replace(month=1, day=1), d_final)
        ]
        for y in range(d_inicial.year+1, d_final.year):
            year_type = "leap" if calendar.isleap(y) else "normal"
            temp = self.tiles[year_type].copy()
            delta_time = \
                datetime.datetime(year=y, month=1, day=1) -\
                datetime.datetime(year=self.years[year_type], month=1, day=1)
            temp.index += delta_time
            years.append(temp)
        return pd.concat(years).sort_index()


# -

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..') # to import modules from parent folder
    
    from db_api import Local
    local = Local()
    
    ena_ree_mean = local.get_table("ampere_climatologia_ena_ree").set_index("ree\mes")
    ena_ree_mean.loc["SIN"] = ena_ree_mean.sum()
    a = Climatologia(ena_ree_mean)
    a(datetime.datetime(1997, 12, 20), datetime.datetime(2001, 1, 10)).plot(figsize=(20, 10))
