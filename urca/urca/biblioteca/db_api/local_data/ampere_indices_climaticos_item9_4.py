import pandas as pd
import os

from .abstract_data import AbstractData


class AmpereIndiceMensalAMO(AbstractData):
    DATA_DICT = {
        "data": "data",
        "AMOm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "AMOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.4-indices_climaticos"
    name = "ampere_indice_AMO_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "AMO", "MENSAL")#.astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_df(cls, data, f_variavel):
        data["day"] = 1
        data["data"] = pd.to_datetime(data[["Year", "Month", "day"]].rename(columns={"Year": 'year', "Month": 'month'}))
        data = data.rename(columns={"Data": f_variavel})
        #print(data.columns)
        return data[["data", f_variavel]]
    
    @classmethod
    def _get_indice(cls, dir_path, indice, periodo):
        data = []
        for filename in os.listdir(dir_path):
            f_variavel = filename.split("_")[1]
            if f_variavel != indice:
                continue
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(dir_path, filename), decimal='.', sep=",")
                data.append(cls._process_df(df, f_variavel))
        return pd.concat(data, axis=0)#.reset_index()


# +

class AmpereIndiceMensalPDO(AmpereIndiceMensalAMO):
    DATA_DICT = {
        "data": "data",
        "PDOm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "PDOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.4-indices_climaticos"
    name = "ampere_indice_PDO_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "PDO", "MENSAL")#.astype(cls.DATA_TYPES_DICT)



# +

class AmpereIndiceMensalTNA(AmpereIndiceMensalAMO):
    DATA_DICT = {
        "data": "data",
        "PDOm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "PDOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.4-indices_climaticos"
    name = "ampere_indice_TNA_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "TNA", "MENSAL")#.astype(cls.DATA_TYPES_DICT)



# +

class AmpereIndiceMensalTSA(AmpereIndiceMensalAMO):
    DATA_DICT = {
        "data": "data",
        "PDOm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "PDOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.4-indices_climaticos"
    name = "ampere_indice_TSA_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "TSA", "MENSAL")#.astype(cls.DATA_TYPES_DICT)



# +

class AmpereIndiceMensalNINOS(AmpereIndiceMensalAMO):
    DATA_DICT = {
        "data": "data",
        "NINOS": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "PDOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.4-indices_climaticos"
    name = "ampere_indice_NINOS_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_NINOS(path, "NINOS", "MENSAL")#.astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_df(cls, df, f_variavel):
        df["day"] = 1
        df['data'] = pd.to_datetime(df[["YR", "MON", "day"]].rename(columns={"YR": 'year', "MON": 'month'}))
        df = df.drop(["YR", "MON", "day" ], axis=1)
        lista = list(df.columns)[:-1]
        lista = ["data"] + lista
        df = df[lista]
        return df#[["data", f_variavel]]
    
    @classmethod
    def _get_NINOS(cls, dir_path, indice, periodo):
        data = []
        for filename in os.listdir(dir_path):
            f_variavel = filename.split("_")[1]
            if f_variavel != indice:
                continue
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(dir_path, filename), decimal='.', sep=";")
                data.append(cls._process_df(df, f_variavel))
        return pd.concat(data, axis=0)#.reset_index()
