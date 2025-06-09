import pandas as pd
import os

from .abstract_data import AbstractData
''' Histórico de previsão climática da precipitação'''


class AmperePrevClimaticaREE(AbstractData):
    DATA_DICT = {
        "data": "data de previsão M+0",
        "M+1": "previsão",
        "M+2": "previsão",
        "M+3": "previsão",
        "M+4": "previsão",
        "M+5": "previsão",
        "M+6": "previsão",
        "M+7": "previsão",
        "M+8": "previsão",
        "M+9": "previsão",
        "M+10": "previsão",
        "M+11": "previsão",
        "cod_rees": "",
        "modelo": "",
        "nome_rees": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "M+1": "float64",
        "M+2": "float64",
        "M+3": "float64",
        "M+4": "float64",
        "M+5": "float64",
        "M+6": "float64",
        "M+7": "float64",
        "M+8": "float64",
        "M+9": "float64",
        "M+10": "float64",
        "M+11": "float64",
        "cod_rees": "object",
        "modelo": "object",
        "nome_rees": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item3-Previsão_climática/resultados"
    name = "ampere_previsao_climatica_rees"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if modelo != "_MERGE":
                    path_temp = os.path.join(path, modelo, "rees")
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    for df, cod, name in raw:
                        df["cod_rees"] = cod[len("ree"):]
                        df["modelo"] = modelo
                        df["nome_rees"] = name
                        data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _load_data_common(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                n_col = 12
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None, names=range(n_col)).dropna(how='all')
                df = df.rename({i: f"M+{i}" for i in range (1, len(df.columns))}, axis='columns')
                df = df.rename({0: 'data'}, axis='columns') 
                df['data'] = pd.to_datetime(df['data'], format="%Y%m")
                cod, name, _ = filename.split("_", 2)
                data.append((df, cod, name))
        return data


class AmperePrevClimaticaBacia(AmperePrevClimaticaREE):
    DATA_DICT = {
        "data": "data",
        "M+1": "previsão",
        "M+2": "previsão",
        "M+3": "previsão",
        "M+4": "previsão",
        "M+5": "previsão",
        "M+6": "previsão",
        "M+7": "previsão",
        "M+8": "previsão",
        "M+9": "previsão",
        "M+10": "previsão",
        "M+11": "previsão",
        "cod_rees": "",
        "modelo": "",
        "nome_rees": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "M+1": "float64",
        "M+2": "float64",
        "M+3": "float64",
        "M+4": "float64",
        "M+5": "float64",
        "M+6": "float64",
        "M+7": "float64",
        "M+8": "float64",
        "M+9": "float64",
        "M+10": "float64",
        "M+11": "float64",
        "cod_rees": "object",
        "modelo": "object",
        "nome_rees": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item3-Previsão_climática/resultados"
    name = "ampere_previsao_climatica_bacias"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if modelo != "_MERGE":
                    path_temp = os.path.join(path, modelo, "bacias")
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    for df, cod, name in raw:
                        df["cod_bacia"] = cod[len("bacia"):]
                        df["modelo"] = modelo
                        df["nome_bacia"] = name
                        data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)
