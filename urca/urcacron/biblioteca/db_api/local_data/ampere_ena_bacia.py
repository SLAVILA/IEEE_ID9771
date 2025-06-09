import pandas as pd
import os

from .abstract_data import AbstractData
''' Energia natural afluente histórica '''


class AmpereEnaHistBacia(AbstractData):
    DATA_DICT = {
        "data": "O histórico de ENA diária foi calculado com base nos dados de \
        vazão natural para cada posto hidráulico de interesse do SIN e na \
        produtibilidade histórica associada a cada um deles, considerando o \
        horizonte de 01/01/2000 a 30/04/2023.",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "GRANDE": "float64",
        "PARANAÍBA": "float64",
        "ALTO PARANÁ": "float64",
        "PARANAPANEMA": "float64",
        "IGUAÇU": "float64",
        "URUGUAI": "float64",
        "TAQUARI-ANTAS": "float64",
        "ITAJAÍ-AÇU": "float64",
        "TIETÊ": "float64",
        "JACUÍ": "float64",
        "CAPIVARI": "float64",
        "PARAÍBA DO SUL": "float64",
        "DOCE": "float64",
        "MADEIRA": "float64",
        "BAIXO PARANÁ": "float64",
        "SÃO FRANCISCO": "float64",
        "JEQUITINHONHA": "float64",
        "PARNAÍBA": "float64",
        "TOCANTINS": "float64",
        "ITABAPOANA": "float64",
        "ARAGUARI": "float64",
        "TELES PIRES": "float64",
        "PARAGUAI": "float64",
        "PARAGUAÇU": "float64",
        "UATUAMA": "float64",
        "CURUA-UNA   ": "float64",
        "MUCURI": "float64",
        "XINGU": "float64",
        "JARI": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/bacias"
    name = "ampere_ENA_hist_bacia"

    @classmethod
    def _get_data(cls, path):
        return AmpereEnaHistBacia._get_ena(path, "historica").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _get_ena(cls, dir_path, tipo):
        data = []
        for arquivo in os.listdir(dir_path):
            f_tipo = arquivo.split("_")[-1][:-4]
            if f_tipo != tipo:
                continue
            temp = pd.read_csv(os.path.join(dir_path, arquivo), sep=";", decimal=",", header=0).rename(columns={"Data": "data"})
            temp['data'] = pd.to_datetime(temp['data'], format="%d/%m/%Y")
            data.append(temp.set_index("data"))

        return pd.concat(data, axis=0).reset_index()

class AmpereEnaAtualBacia(AmpereEnaHistBacia):
    DATA_DICT = {
        #fazer
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "GRANDE": "float64",
        "PARANAÍBA": "float64",
        "ALTO PARANÁ": "float64",
        "PARANAPANEMA": "float64",
        "IGUAÇU": "float64",
        "URUGUAI": "float64",
        "TAQUARI-ANTAS": "float64",
        "ITAJAÍ-AÇU": "float64",
        "TIETÊ": "float64",
        "JACUÍ": "float64",
        "CAPIVARI": "float64",
        "PARAÍBA DO SUL": "float64",
        "DOCE": "float64",
        "MADEIRA": "float64",
        "BAIXO PARANÁ": "float64",
        "SÃO FRANCISCO": "float64",
        "JEQUITINHONHA": "float64",
        "PARNAÍBA": "float64",
        "TOCANTINS": "float64",
        "ITABAPOANA": "float64",
        "ARAGUARI": "float64",
        "TELES PIRES": "float64",
        "PARAGUAI": "float64",
        "PARAGUAÇU": "float64",
        "UATUAMA": "float64",
        "CURUA-UNA   ": "float64",
        "MUCURI": "float64",
        "XINGU": "float64",
        "JARI": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/bacias"
    name = "ampere_ENA_atual_bacia"

    @classmethod
    def _get_data(cls, path):
        return super()._get_ena(path, "atual").astype(cls.DATA_TYPES_DICT)
