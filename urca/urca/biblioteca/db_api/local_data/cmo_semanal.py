import pandas as pd
import os

from .abstract_data import AbstractData


class CMOSemanal(AbstractData):
    DATA_DICT = {  # https://dados.ons.org.br/dataset/cmo-semanal
        "id_subsistema": "identificador do subsistema",
        "nom_subsistema": "nome do subsistema",
        "din_instante": "Data",
        "val_cmomediasemanal": "CMO Médio Semanal, em R$/MW ", 
        "val_cmoleve": "CMO no patamar de carga leve, em R$/MWh",
        "val_cmomedia": "CMO no patamar de carga média, em R$/MWh",
        "val_cmopesada": "CMO no patamar de carga pesada, em R$/MWh"
    }
    DATA_TYPES_DICT = {
        "id_subsistema": "object",
        "nom_subsistema": "object",
        "din_instante": "datetime64[ns]",
        "val_cmomediasemanal": "float64",
        "val_cmoleve": "float64",
        "val_cmomedia": "float64",
        "val_cmopesada": "float64"
    }
    DEFAULT_PATH = "Dados/raw/cmo_semanal"
    name = "cmo_semanal"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal='.')
                data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
