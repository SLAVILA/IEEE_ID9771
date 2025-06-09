import pandas as pd
import os

from .abstract_data import AbstractData


class PLDMensal(AbstractData):
    DATA_DICT = {  # https://www.ccee.org.br/precos/painel-precos
        "data": "mes e ano",
        "submercado": "submercado",
        "PLD_mensal_medio": "PLD_mensal_medio",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "submercado": "object",
        "PLD_mensal_medio": "float",
    }
    DEFAULT_PATH = "Dados/raw/PLD_mensal"
    name = "PLD_mensal"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep='\t')
                df["data"] = pd.to_datetime(df['data'], format="%d/%m/%y")
                df["PLD_mensal_medio"] = df["PLD_mensal_medio"].apply(lambda x: x.replace(',','.'))
                data.append(df)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
