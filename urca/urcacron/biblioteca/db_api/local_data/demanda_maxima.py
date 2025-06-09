import pandas as pd
import os

from .abstract_data import AbstractData


class DemandaMaxima(AbstractData):
    DATA_DICT = {  # https://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/demanda_maxima.aspx
        "Din Instante": "Data",
        "Subsistema": "Subsistema da demanda máxima ou 'SIN' para o valor total",
        "Demanda Maxima": "Demanda máxima no dia'"
    }
    DATA_TYPES_DICT = {
        "Din Instante": "datetime64[ns]",
        "Subsistema": "object",
        "Demanda Maxima": "float64"
    }
    DEFAULT_PATH = "Dados/raw/demanda_maxima"
    name = "demanda_maxima"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep="\t", decimal='.', skiprows=[0], encoding="utf-16")
                df = df.drop(df.columns[: 3], axis=1)
                df["Din Instante"] = pd.to_datetime(df["Din Instante"], format="%d/%m/%Y %H:%M:%S").dt.date

                def filter_row(row):
                    data = row.dropna().to_numpy()
                    if len(data) != 1:
                        raise ValueError("A linha não tem exatamente 1 valor válido")
                    return data[0]

                cols = df.columns[2:]
                df['Demanda Maxima'] = df[cols].apply(filter_row, axis=1)
                df = df.drop(cols, axis=1)

                data.append(df)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
