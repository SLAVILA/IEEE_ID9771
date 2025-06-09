import pandas as pd
import os

from .abstract_data import AbstractData


# +

class AmpereIndiceMensalPrevisao24TNA(AbstractData):
    DATA_DICT = {
        "data": "data",
        "TNAm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "TNAm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.3-indices_climaticos/"
    name = "ampere_indice_TNA_mensal_previsao24"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo), "TNA", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _get_indice(cls, dir_path, indice, modelo):
        data = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                #print(filename)
                f_variavel = filename.split("_")[-1][0:-4]
                if indice == "NINO":
                    nino = f_variavel
                    f_variavel = (f_variavel[0:4]).upper()
                if f_variavel != indice:
                    continue
                #print(filename, nino, f_variavel, indice)
                #print(modelo)
                if modelo != "LIM":
                    df = pd.read_csv(os.path.join(dir_path, filename), decimal='.', sep=";")
                else:
                    df = pd.read_csv(os.path.join(dir_path, filename), decimal='.', sep=",")
                df["Date"] = pd.to_datetime(df["Date"], format='%Y%m')
                df = df.rename(columns={"Date": "data"})
                if indice == "NINO":
                    df["NINO"] = nino
                data.append(df)
        data = pd.concat(data)
        return data


# +

class AmpereIndiceMensalPrevisao24TSA(AmpereIndiceMensalPrevisao24TNA):
    DATA_DICT = {
        "data": "data",
        "TSAm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "TSAm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.3-indices_climaticos/"
    name = "ampere_indice_TSA_mensal_previsao24"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo), "TSA", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)



# +

class AmpereIndiceMensalPrevisao24NINOS(AmpereIndiceMensalPrevisao24TNA):
    DATA_DICT = {
        "data": "data",
        "NINOS": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "NINOS": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.3-indices_climaticos/"
    name = "ampere_indice_NINOS_mensal_previsao24"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo), "NINO", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)

