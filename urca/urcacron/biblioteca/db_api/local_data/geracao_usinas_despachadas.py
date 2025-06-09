import pandas as pd
import datetime
import os

from .abstract_data import AbstractData


class GeracaoUsinasDespachadas(AbstractData):
    # https://www.ons.org.br/Paginas/resultados-da-operacao/historico-da-operacao/geracao_energia.aspx
    DATA_DICT = {
        "Data": "Data da amostra",
        "Subsistema": "Subsistema do SIM",
        "Tipo": "Tipo de usina",
        "Geração": "Geração média em MW",
    }
    DATA_TYPES_DICT = {
        "Data": "datetime64[ns]",
        "Subsistema": "object",
        "Tipo": "object",
        "Geração": "float64"
    }
    DEFAULT_PATH = "Dados/raw/geracao_usinas_despachadas"
    name = "geracao_usinas_despachadas"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".xlsx"):
                temp = pd.read_excel(os.path.join(path, filename), skiprows=[0])
                temp = temp[~temp["Data Dica Comp"].isna()]

                def filter_row(row):
                    data = row.dropna().to_numpy()
                    if len(data) != 1:
                        raise ValueError("A linha não tem exatamente 1 valor válido")
                    return data[0]

                df = pd.DataFrame()
                df["Data"] = temp["Data Dica Comp"].apply(
                    lambda d: datetime.datetime(*[int(n) for n in d.split("/")][::-1])
                )
                df["Subsistema"] = temp["Selecione Comparar GE Comp 3.1"]
                # arquivos tem o nome Comparativo Geração de Energia - [Tipo] [Ano].xlsx
                df["Tipo"] = filename.split(" ")[5]
                df["Geração"] = temp[temp.columns[9:]].apply(filter_row, axis=1)

                data.append(df)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)
