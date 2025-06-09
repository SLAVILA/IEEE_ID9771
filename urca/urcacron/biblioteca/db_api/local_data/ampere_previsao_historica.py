import pandas as pd
import os

from .abstract_data import AbstractData
''' Previsão histórica de precipitação (diaria)'''


class AmperePrevisaoHistoricaREE(AbstractData):
    DATA_DICT = {
        "data": "data de previsão D+0 (diaria)",
        "D+1": "previsão",
        "D+2": "previsão",
        "D+3": "previsão",
        "D+4": "previsão",
        "D+5": "previsão",
        "D+6": "previsão",
        "D+7": "previsão",
        "D+8": "previsão",
        "D+9": "previsão",
        "D+10": "previsão",
        "D+11": "previsão",
        "D+12": "previsão",
        "D+13": "previsão",
        "D+14": "previsão",
        "cod_rees": "object",
        "modelo": "object",
        "nome_rees": "object",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "D+1": "float64",
        "D+2": "float64",
        "D+3": "float64",
        "D+4": "float64",
        "D+5": "float64",
        "D+6": "float64",
        "D+7": "float64",
        "D+8": "float64",
        "D+9": "float64",
        "D+10": "float64",
        "D+11": "float64",
        "D+12": "float64",
        "D+13": "float64",
        "D+14": "float64",
        "cod_rees": "object",
        "modelo": "object",
        "nome_rees": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item2-previsao_historica"
    name = "previsao_historica_rees"

    @classmethod
    def _get_data(cls, path):            
        from dateutil.parser import ParserError
        data = []
        for modelo in os.listdir(path):
            #print(modelo)
            if os.path.isdir(os.path.join(path, modelo)):
                path_temp = os.path.join(path, modelo, "rees")
                #print(path_temp)
                if modelo == "cfsv2":
                    raw = cls._load_data_common(path_temp, range(595))
                else:
                    raw = cls._load_data_common(path_temp)
                for df, cod, name in raw:
                    df["cod_rees"] = cod[len("ree"):]
                    df["modelo"] = modelo
                    df["nome_rees"] = name
                    data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _load_data_common(cls, path, names=None):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None, names=names)
                df = df.rename({i: f"D+{i}" for i in range (1, len(df.columns))}, axis='columns')
                df = df.rename({0: 'data'}, axis='columns')
                try:
                    df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")
                except:
                    df['data'] = pd.to_datetime(df['data'], format="%Y/%m/%d")
                df['data'] = df['data'].dt.date
                cod, name, _ = filename.split("_", 2)
                data.append((df, cod, name))
        return data

class AmperePrevisaoHistoricaBacias(AmperePrevisaoHistoricaREE):
    DATA_DICT = {
        "data": "data",
        "D+1": "previsão",
        "D+2": "previsão",
        "D+3": "previsão",
        "D+4": "previsão",
        "D+5": "previsão",
        "D+6": "previsão",
        "D+7": "previsão",
        "D+8": "previsão",
        "D+9": "previsão",
        "D+10": "previsão",
        "D+11": "previsão",
        "D+12": "previsão",
        "D+13": "previsão",
        "D+14": "previsão",
        "cod_bacia": "object",
        "modelo": "object",
        "nome_bacia": "object",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "D+1": "float64",
        "D+2": "float64",
        "D+3": "float64",
        "D+4": "float64",
        "D+5": "float64",
        "D+6": "float64",
        "D+7": "float64",
        "D+8": "float64",
        "D+9": "float64",
        "D+10": "float64",
        "D+11": "float64",
        "D+12": "float64",
        "D+13": "float64",
        "D+14": "float64",
        "cod_bacia": "object",
        "modelo": "object",
        "nome_bacia": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item2-previsao_historica"
    name = "previsao_historica_bacias"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                path_temp= os.path.join(path, modelo, "bacias")
                #print(path_temp)
                if modelo == "cfsv2":
                    raw = cls._load_data_common(path_temp, range(595))
                else:
                    raw = cls._load_data_common(path_temp)
                for df, cod, name in raw:
                    df["cod_bacia"] = cod[len("bacia"):]
                    df["modelo"] = modelo
                    df["nome_bacia"] = name
                    data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
