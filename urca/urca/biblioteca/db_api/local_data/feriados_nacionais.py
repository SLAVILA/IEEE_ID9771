import pandas as pd
import os

from .abstract_data import AbstractData


class FeriadosNacionais(AbstractData):
    DATA_DICT = {  # https://www.ccee.org.br/web/guest/precos/conceitos-precos
        "Data": "Ano, Mês e dia",
        "Feriado": "Descrição do Feriado",
    }
    DATA_TYPES_DICT = {
        "Data": "datetime64[ns]",
        "Feriado": "object",
    }
    DEFAULT_PATH = "Dados/raw/feriados_nacionais"
    name = "feriados_nacionais"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=',', encoding='Windows-1252', usecols=["Data", "Feriado"])
                df['Data'] = pd.to_datetime(df['Data'], format="%d/%m/%Y")
                data.append(df)
        return pd.concat(data).astype(cls.DATA_TYPES_DICT).sort_values(by="Data")

