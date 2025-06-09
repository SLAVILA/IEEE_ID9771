import pandas as pd
import os

from .abstract_data import AbstractData


class IntercambioNacional(AbstractData):
    # https://dados.ons.org.br/dataset/intercambio-nacional
    DATA_DICT = {
        "din_instante": "Data/hora (início do período de agregação)",
        "id_subsistema_origem": "Código do Subsistema de Origem",
        "nom_subsistema_origem": "Nome do Subsistema de Origem ",
        "id_subsistema_destino": "Código do Subsistema de Destino",
        "nom_subsistema_destino": "Nome do Subsistema de Destino",
        "val_intercambiomwmed": "Intercâmbio verificado em base horária, representando a soma das medidas de fluxo de potência ativa nas linhas de transmissão de fronteira entre os subsistemas em MWmed",
        "orig_dest": "Código do Subsistema de Origem -> Código do Subsistema de Destino",
    }
    DATA_TYPES_DICT = {
        "din_instante": "datetime64[ns]",
        "id_subsistema_origem": "object",
        "nom_subsistema_origem": "object",
        "id_subsistema_destino": "object",
        "nom_subsistema_destino": "object",
        "val_intercambiomwmed": "float64",
        "orig_dest": "object",
    }
    DEFAULT_PATH = "Dados/raw/intercambio_nacional"
    name = "intercambio_nacional"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal='.')
                data.append(df)

        data_f = pd.concat(data)
        data_f["orig_dest"] = data_f.id_subsistema_origem.str.cat(data_f.id_subsistema_destino, sep ="->")
        return data_f.astype(cls.DATA_TYPES_DICT)
 
