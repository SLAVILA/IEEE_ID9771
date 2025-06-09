import pandas as pd
import os

from .abstract_data import AbstractData


class EnergiaVertidaTurbinavel(AbstractData):
    DATA_DICT = {  # https://dados.ons.org.br/dataset/energia-vertida-turbinavel
        "id_subsistema": "identificador do subsistema",
        "nom_subsistema": "nome do subsistema",
        "nom_bacia": "nome da bacia hidroenergética",
        "nom_rio": "nome do rio",
        "nom_agente": "nome do agente",
        "nom_reservatorio": "nome do reservatorio",
        "cod_usina": "Código da usina nos modelos de otimização",
        "din_instante": "Data",
        "val_geracao": "Valor da Geração, em MWmed",
        "val_disponibilidade": "Valor de disponibilidade, em MWmed ",
        "val_vazaoturbinada": "Vazão turbinada, em m3/s",
        "val_vazaovertida": "Vazão vertida, em m3/s",
        "val_vazaovertidanaoturbinavel": "Vazão vertida não turbinável, em m3/s",
        "val_produtividade": "Valor da produtividade, em MW/(m3/s)",
        "val_folgadegeracao": "Valor da folga de geração, em MWmed",
        "val_energiavertida": "Valor da energia vertida, em MWmed",
        "val_vazaovertidaturbinavel": "Vazão vertida turbinável, em m3/s",
        "val_energiavertidaturbinavel": "Valor da energia vertida turbinável, em MWmed",
    }
    DATA_TYPES_DICT = {
        "id_subsistema": "object",
        "nom_subsistema": "object",
        "nom_bacia": "object",
        "nom_rio": "object",
        "nom_agente": "object",
        "nom_reservatorio": "object",
        "cod_usina": "Int64",
        "din_instante": "datetime64[ns]",
        "val_geracao": "float64",
        "val_disponibilidade": "float64",
        "val_vazaoturbinada": "float64",
        "val_vazaovertida": "float64",
        "val_vazaovertidanaoturbinavel": "float64",
        "val_produtividade": "float64",
        "val_folgadegeracao": "float64",
        "val_energiavertida": "float64",
        "val_vazaovertidaturbinavel": "float64",
        "val_energiavertidaturbinavel": "float64",
    }
    DEFAULT_PATH = "Dados/raw/energia_vertida_turbinavel"
    name = "energia_vertida_turbinavel"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), decimal='.', sep=";")
                data.append(df)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
