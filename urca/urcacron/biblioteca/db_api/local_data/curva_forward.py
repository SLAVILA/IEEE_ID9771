import pandas as pd
import os

from .abstract_data import AbstractData


class CurvaForward(AbstractData):
    DATA_DICT = {  # Dados fornecidos pela URCA
        "Mês Vértice": "Mês e ano do produto",
        "Valor": "Preço do produto na 'Data Referência'",
        "Data Referência": "Data da amostra"
    }
    DATA_TYPES_DICT = {
        "Mês Vértice": "datetime64[ns]",
        "Valor": "float64",
        "Data Referência": "datetime64[ns]"
    }
    DEFAULT_PATH = "Dados/raw/curva_forward"
    name = "curva_forward"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".xlsx"):
                sheets = pd.read_excel(os.path.join(path, filename), sheet_name=None, usecols=[1, 2, 3], decimal=',')

                #  Algumas colunas estão com as datas em strings em pt
                # está função traduz para inglês
                translation = {
                    "jan": "jan", "fev": "feb", "mar": "mar", "abr": "apr",
                    "mai": "may", "jun": "jun", "jul": "jul", "ago": "aug",
                    "set": "sep", "out": "oct", "nov": "nov", "dez": "dec"
                }
                def convert(a):
                    if isinstance(a, str):
                        month, year = a.split("/")
                        try:
                            month = translation[month.lower()]
                            a = month+"/"+year
                        except KeyError:
                            pass
                    return a

                # Faz a conversão das colunas com datas
                for key, df in sheets.items():
                    for column in ["Mês Vértice", "Data Referência"]:
                        df[column] = pd.to_datetime(df[column].apply(convert)).dt.date
                
                data.append(pd.concat(sheets.values()))

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)


