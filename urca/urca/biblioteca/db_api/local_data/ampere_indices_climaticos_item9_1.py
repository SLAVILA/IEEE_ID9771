import pandas as pd
import os

from .abstract_data import AbstractData


class AmpereIndiceMensalComplementoAMO(AbstractData):
    DATA_DICT = {
        "data": "data",
        "AMOm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "AMOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.1-indices_climaticos/"
    name = "ampere_indice_AMO_mensal_previsao"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo),"AMO", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _get_indice(cls, dir_path, indice, modelo):
        data = []
        for filename in os.listdir(dir_path):
            f_variavel = filename.split("_")[0]
            if f_variavel == "NINO":
                nino = filename.split("_")[1]
            if f_variavel != indice:
                continue
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(dir_path, filename), decimal='.', sep=";")
                df["data_rodada"] = pd.to_datetime(df["data_rodada"], format='%Y%m')
                df = df.rename(columns={"data_rodada": "data"})
                data.append(df)
                print(data)
        data = pd.concat(data)
        if indice == "NINO":
            return (data, nino)
        else:
            return data


class AmpereIndiceMensalComplementoPDO(AmpereIndiceMensalComplementoAMO):
    DATA_DICT = {
        "data": "data",
        "PDOm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "PDOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.1-indices_climaticos/"
    name = "ampere_indice_PDO_mensal_previsao"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo), "PDO", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)


# +

class AmpereIndiceMensalComplementoTNA(AmpereIndiceMensalComplementoAMO):
    DATA_DICT = {
        "data": "data",
        "TNAm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "TNAm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.1-indices_climaticos/"
    name = "ampere_indice_TNA_mensal_previsao"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo), "TNA", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)


# +

class AmpereIndiceMensalComplementoTSA(AmpereIndiceMensalComplementoAMO):
    DATA_DICT = {
        "data": "data",
        "TSAm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "TSAm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.1-indices_climaticos/"
    name = "ampere_indice_TSA_mensal_previsao"

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

class AmpereIndiceMensalSAD(AmpereIndiceMensalComplementoAMO):
    DATA_DICT = {
        "data": "data",
        "SADm": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "SADm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.1-indices_climaticos/"
    name = "ampere_indice_SAD_mensal"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                raw = cls._get_indice(os.path.join(path, modelo), "SAD", modelo)
                raw['modelo'] = modelo
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)



# +

class AmpereIndiceMensalComplementoNINOS(AmpereIndiceMensalComplementoAMO):
    DATA_DICT = {
        "data": "data",
        "NINOS": " ",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "PDOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.1-indices_climaticos/"
    name = "ampere_indice_NINOS_mensal_previsao"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            print(modelo)
            if os.path.isdir(os.path.join(path, modelo)):
                raw, nino = cls._get_indice(os.path.join(path, modelo), "NINO", modelo)
                raw['modelo'] = modelo
                raw['NINO'] = nino
                data.append(raw)
        return pd.concat(data)  # .astype(cls.DATA_TYPES_DICT)

