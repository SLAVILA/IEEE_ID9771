import pandas as pd
import os

from .abstract_data import AbstractData
''' Histórico da precipitação observada '''


class AmperePrecipitacaoHistoricaREE(AbstractData):
    DATA_DICT = {
        "data": "",
        "precipitacao": "",
        "cod_rees": "",
        "nome_rees": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "precipitacao": "float64",
        "cod_rees": "int",
        "nome_rees": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item1-precipitacao_historica/rees"
    name = "ampere_precipitacao_historica_rees"

    @classmethod
    def _get_data(cls, path):
        raw = AmperePrecipitacaoHistoricaREE._load_data_common(path)
        data = []
        for df, cod, name in raw:
            df["cod_rees"] = cod[len("ree"):]
            df["nome_rees"] = name
            data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _load_data_common(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None).rename({0: 'data', 1: 'precipitacao'}, axis='columns')
                cod, name, _ = filename.split("_", 2)
                data.append((df, cod, name))
        return data

class AmperePrecipitacaoHistoricaBacias(AbstractData):
    DATA_DICT = {
        "data": "",
        "precipitacao": "",
        "cod_bacia": "",
        "nome_bacia": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "precipitacao": "float64",
        "cod_bacia": "int",
        "nome_bacia": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item1-precipitacao_historica/bacias"
    name = "ampere_precipitacao_historica_bacias"
    
    @classmethod
    def _get_data(cls, path):
        raw = AmperePrecipitacaoHistoricaREE._load_data_common(path)
        data = []
        for df, cod, name in raw:
            df["cod_bacia"] = cod[len("bacia"):]
            df["nome_bacia"] = name
            data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
