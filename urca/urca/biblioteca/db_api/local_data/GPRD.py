import pandas as pd
import os

from .abstract_data import AbstractData


class PLDMensal(AbstractData):
    DATA_DICT = {  # https://www.matteoiacoviello.com/gpr.htm
        "data": "mes e ano",
        "GPRD": "geopolitical risk",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "GPRD": "float",
    }
    DEFAULT_PATH = "Dados/raw/GPRD"
    name = "GPRD"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".xls"):
                df = pd.read_excel(os.path.join(path, filename))
                df.to_csv('.csv', encoding='utf-8')
                df["data"] = pd.to_datetime(df['data'], format="%d/%m/%y")
                df["PLD_mensal_medio"] = df["PLD_mensal_medio"].apply(lambda x: x.replace(',','.'))
                data.append(df)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)