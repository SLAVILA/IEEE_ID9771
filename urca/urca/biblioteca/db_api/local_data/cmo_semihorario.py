import pandas as pd
import os

from .abstract_data import AbstractData


class CMOSemihorario(AbstractData):
    DATA_DICT = {  # https://dados.ons.org.br/dataset/cmo-semi-horario
        "id_subsistema": "identificador do subsistema",
        "nom_subsistema": "nome do subsistema",
        "din_instante": "Data",
        "val_cmo": "CMO a cada meia hora, em R$/MWh",
    }
    DATA_TYPES_DICT = {
        "id_subsistema": "object",
        "nom_subsistema": "object",
        "din_instante": "datetime64[ns]",
        "val_cmo": "float64",
    }
    DEFAULT_PATH = "Dados/raw/CMO_semihorario_ons"
    name = "cmo_semihorario"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal='.')
                data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
