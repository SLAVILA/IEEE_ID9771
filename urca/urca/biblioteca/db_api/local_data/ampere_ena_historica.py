import pandas as pd
import os

from .abstract_data import AbstractData
''' Energia natural afluente histórica '''


class AmpereEnaHistMensal(AbstractData):
    DATA_DICT = {
    "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/mensal"
    name = "AmpereEnaHistMensal"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                print(filename)
                path_temp = os.path.join(path, filename)
                print(path_temp)
                raw = cls._load_data_common(path_temp, filename)
                for df, name in raw:
                    df["tipo"] = name
                    df["modelo"] = "historica_mensal"
                    data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _load_data_common(cls, path, filename):
        aaa = filename.split("_")
        name = aaa[-4]
        data = []
        Meses_dict = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}
        df = pd.read_csv(path, sep=";", decimal=",", encoding='windows-1252')#, skiprows=[0])
        print(df.columns)
        print(name)
        if name != "bacias":
            df = df.rename({'data\sub': 'data'}, axis='columns')
        else:
            df = df.rename(columns={list(df.columns)[0]: 'data'})
            print("aaaa")

        df.index.name = 'data'
        datas = []
        print(df.columns)
        for row in range(len(df)):
            datas.append(df['data'][row].split('/'))
        datas_ok = []
        for mes, ano in datas:
            datas_ok.append(str(2000+int(ano)if int(ano) < 31 else 1931+int(ano)-31)+"/"+str(Meses_dict[mes]))
        df['data'] = pd.to_datetime(datas_ok, format="%Y/%m")
        
        data.append((df, name))
        return data