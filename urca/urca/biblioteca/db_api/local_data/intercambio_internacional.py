import pandas as pd
import os

from .abstract_data import AbstractData


class IntercambioInternacional(AbstractData):
    # https://dados.ons.org.br/dataset/intercambio-nacional
    DATA_DICT = {
        "din_instante": "Data/hora (início do período de agregação)",
        "nom_paisdestino": "Nome do país de destino da energia",
        "val_intercambiomwmed": "Intercâmbio verificado em base horária, representando a soma das medidas de fluxo de potência ativa nas conversoras de frequência de fronteira entre os dois países medido, em MWmed. Dados positivos indicam exportação de energia do Brasil para outros países; dados negativos indicam importação de energia do Brasil de outros países.",
    }
    DATA_TYPES_DICT = {
        "din_instante": "datetime64[ns]",
        "nom_paisdestino": "object",
        "val_intercambiomwmed": "float64",
    }
    DEFAULT_PATH = "Dados/raw/intercambio_internacional"
    name = "intercambio_internacional"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal='.')
                data.append(df)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
