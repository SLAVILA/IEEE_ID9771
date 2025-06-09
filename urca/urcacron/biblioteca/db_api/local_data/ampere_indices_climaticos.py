import pandas as pd
import os

from .abstract_data import AbstractData


class AmpereIndiceDiarioAAO(AbstractData):
    DATA_DICT = {
        "data": "data",
        "AAOd": "Oscilação Antártica",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "AAOd": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_AOO_diario"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "AAO", "DIARIO").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0, encoding='Windows-1252')
        data["data"] = pd.to_datetime(data[["Ano", "Mês", "Dia"]].rename(columns={"Ano": 'year', "Mês": 'month', "Dia": 'day'}))
        return data[["data", "AAOd"]]
    
    @classmethod
    def _get_indice(cls, dir_path, indice, periodo):
        data = []
        for arquivo in os.listdir(dir_path):
            try:
                f_variavel = arquivo.split("_")[1]
            except:
                f_variavel = arquivo.split(".")[0]
            if f_variavel != indice:
                continue
            f_tipo = arquivo.split("_")[-1][:-4]
            if f_tipo != periodo:
                continue
            #print(f_tipo)
            data.append(cls._process_csv(os.path.join(dir_path, arquivo)))
        return pd.concat(data, axis=0)#.reset_index()


class AmpereIndiceMensalAAO(AmpereIndiceDiarioAAO):
    DATA_DICT = {
        #fazer
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "AAOm": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_AOO_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "AAO", "MENSAL").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0)
        data["Dia"] = 1
        data["data"] = pd.to_datetime(data[["Ano", "Mês", "Dia"]].rename(columns={"Ano": 'year', "Mês": 'month', "Dia": 'day'}))
        return data[["data", "AAOm"]]


class AmpereIndiceMensalIOD(AmpereIndiceDiarioAAO):
    DATA_DICT = {
        #fazer
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "IODm": "float",
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_IOD_mensal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "IOD", "MENSAL").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0, na_values='-').dropna()
        data["Dia"] = 1
        data["data"] = pd.to_datetime(data[["Ano", "Mês", "Dia"]].rename(columns={"Ano": 'year', "Mês": 'month', "Dia": 'day'}))
        return data[["data", "IODm"]]


class AmpereIndiceSemanalIOD(AmpereIndiceDiarioAAO):
    DATA_DICT = {
        "data": "",
        "IODs": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        #"data_inicial": "datetime64[ns]",
        #"data_final": "datetime64[ns]",
        "IODs": "float64",

    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_IOD_semanal"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "IOD", "SEMANAL").astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0)
        data["Dia"] = 1
        data["data"] = pd.to_datetime(data["Data inicial"], format="%Y%m%d")
        #data["data_final"] = pd.to_datetime(data["Data final"], format="%Y%m%d")
        return data[["data", "IODs"]]#["data_inicial", "data_final", "IODs"]]

class AmpereIndiceDiarioMJO(AmpereIndiceDiarioAAO):
    DATA_DICT = {
        "data": "",
        "": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "RMM1": "float64",
        "RMM2": "float64",
        "Fase": "int64",
        "Amplitude": "float64",
        
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_MJO_diario"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "MJO", "DIARIO").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0)
        #data["Dia"] = 1
        data["data"] = pd.to_datetime(data[["Ano", "Mês", "Dia"]].rename(columns={"Ano": 'year', "Mês": 'month', "Dia": 'day'}))
        return data[["data", "RMM1", "RMM2", "Fase", "Amplitude"]]


class AmpereIndiceMensalONI(AmpereIndiceDiarioAAO):
    DATA_DICT = {
        #fazer
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "ONIm": "float64",
       
    }
    DEFAULT_PATH = "Dados/raw/ONIm"#"Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_ONI_mensal"

    @classmethod
    def _get_data(cls, path):
        return AmpereIndiceMensalONI._get_indice(path, "ONI", "MENSAL").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0, encoding='Windows-1252')
        data["Dia"] = 1
        data["data"] = pd.to_datetime(data[["Ano", "Mês", "Dia"]].rename(columns={"Ano": 'year', "Mês": 'month', "Dia": 'day'}))
        return data[["data", "ONIm"]]


class AmpereIndiceTrimestralONI(AmpereIndiceDiarioAAO):
    DATA_DICT = {
        #fazer
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "Trimestre": "object",
        "ONIm": "float64",
        "ENSO": "object",
       
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_indice_ONI_trimestral"

    @classmethod
    def _get_data(cls, path):
        return cls._get_indice(path, "ONI", "TRIMESTRAL").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0, encoding='Windows-1252')
        data["Dia"] = 1
        data["data"] = pd.to_datetime(data[["Ano", "Mês", "Dia"]].rename(columns={"Ano": 'year', "Mês": 'month', "Dia": 'day'}))
        data["data"] = (data["data"]-pd.Timedelta(days=1)).dt.to_period('M').dt.to_timestamp()
        return data[["data", "Trimestre", "ONIm", "ENSO"]]