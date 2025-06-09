import pandas as pd
import os

from .abstract_data import AbstractData


class PisoTetoPLD(AbstractData):
    DATA_DICT = {  # https://www.ccee.org.br/web/guest/precos/conceitos-precos
        "Ano": "Ano. Mês e dia foram adicionados por conta do formatado de dado padronizado",
        "PLD_minimo": "PLD_minimo",
        "PLD_maximo": "PLD_maximo",
    }
    DATA_TYPES_DICT = {
        "Ano": "datetime64[ns]",
        "PLD_minimo": "float64",
        "PLD_maximo": "float64",
    }
    DEFAULT_PATH = "Dados/raw/piso_teto_PLD"
    name = "piso_teto_pld"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                piso_teto_pld = pd.read_csv(os.path.join(path, filename), sep=',')
                piso_teto_pld["Ano"] = piso_teto_pld.apply(lambda row: f"{row.Ano}/01/01", axis = 1)
                piso_teto_pld["PLD_minimo"] = piso_teto_pld["PLD_minimo"].str.replace(',', '.')
                piso_teto_pld["PLD_maximo"] = piso_teto_pld["PLD_maximo"].str.replace(',', '.')
                data.append(piso_teto_pld)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
