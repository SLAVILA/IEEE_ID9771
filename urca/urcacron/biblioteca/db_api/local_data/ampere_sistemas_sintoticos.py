import pandas as pd
import os

from .abstract_data import AbstractData


class AmpereZCAS(AbstractData):
    DATA_DICT = {
        "data": "data",
        
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "Posicao_climatologica": "object",
        "JACUI": "int64",
        "TAQUARI-ANTAS": "int64",
        "URUGUAI": "int64",
        "ITAJAI-ACU": "int64",
        "IGUACU": "int64",
        "PARANAPANEMA": "int64",
        "BAIXOPARANA": "int64",
        "CAPIVARI": "int64",
        "TIETE": "int64",
        "PARAIBADOSUL": "int64",
        "GRANDE": "int64",
        "ALTOPARANA": "int64",
        "DOCE": "int64",
        "ITABAPOANA": "int64",
        "SAOFRANCISCO": "int64",
        "MADEIRA": "int64",
        "PARANAIBA": "int64",
        "JEQUITINHONHA": "int64",
        "TOCANTINS": "int64",
        "MUCURI": "int64",
        "PARAGUAI": "int64",
        "TELES-PIRES": "int64",
        "XINGU": "int64",
        "PARAGUACU": "int64",
        "PARNAIBA": "int64",
        "CURUA-UNA": "int64",
        "UATUAMA": "int64",
        "JARI": "int64",
        "ARAGUARI": "int64",
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_ZCAS"

    @classmethod
    def _get_data(cls, path):
        return cls._get_sistema(path, "ZCAS").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _process_csv(cls, csv):
        data = pd.read_csv(csv, sep=";", decimal=",", header=0, index_col=0).rename({"Data": 'data'}, axis='columns')
        data["data"] = pd.to_datetime(data['data'], format='mixed')
        return data
    
    @classmethod
    def _get_sistema(cls, dir_path, sistema):
        data = []
        for arquivo in os.listdir(dir_path):
            f_variavel = arquivo.split(".")[0]
            if f_variavel != sistema:
                continue
            data.append(cls._process_csv(os.path.join(dir_path, arquivo)))
        return pd.concat(data, axis=0)#.reset_index()


class AmpereASAS(AmpereZCAS):
    DATA_DICT = {
        "data": "data",
        
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "JACUI": "int64",
        "TAQUARI-ANTAS": "int64",
        "URUGUAI": "int64",
        "ITAJAI-ACU": "int64",
        "IGUACU": "int64",
        "PARANAPANEMA": "int64",
        "BAIXOPARANA": "int64",
        "CAPIVARI": "int64",
        "TIETE": "int64",
        "PARAIBADOSUL": "int64",
        "GRANDE": "int64",
        "ALTOPARANA": "int64",
        "DOCE": "int64",
        "ITABAPOANA": "int64",
        "SAOFRANCISCO": "int64",
        "MADEIRA": "int64",
        "PARANAIBA": "int64",
        "JEQUITINHONHA": "int64",
        "TOCANTINS": "int64",
        "MUCURI": "int64",
        "PARAGUAI": "int64",
        "TELES-PIRES": "int64",
        "XINGU": "int64",
        "PARAGUACU": "int64",
        "PARNAIBA": "int64",
        "CURUA-UNA": "int64",
        "UATUAMA": "int64",
        "JARI": "int64",
        "ARAGUARI": "int64",
    }
    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_ASAS"

    @classmethod
    def _get_data(cls, path):
        return cls._get_sistema(path, "ASAS").astype(cls.DATA_TYPES_DICT)



class AmpereJBN(AmpereASAS):

    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_JBN"

    @classmethod
    def _get_data(cls, path):
        return cls._get_sistema(path, "JBN").astype(cls.DATA_TYPES_DICT)



class AmpereFrentesFrias(AmpereASAS):

    DEFAULT_PATH = "Dados/Ampere/item8-indices_climaticos"
    name = "ampere_Frentes_frias"

    @classmethod
    def _get_data(cls, path):
        return cls._get_sistema(path, "Frentes_frias")#.astype(cls.DATA_TYPES_DICT)

