import pandas as pd
import os

from .abstract_data import AbstractData


class IndicadorPrevisaoEna(AbstractData):
    DATA_DICT = {  # https://www.ons.org.br/Paginas/sobre-ons/indicadores-ons/indicador_IPE.aspx
        "Ano": "",
        "Mês": "Mes do PMO", # Conferir descricao
        "Região": "Região",
        "Rev. PMO": "Revisao do Programa Mensal da Operacao", 
        "ENA_obs": "Energia Natural Afluente observada",
        "ENA_prev": "Energia Natural Afluente prevista",
        "MLT": "Média de Longo Prazo", # sigla aparece no relatorio MACRO 1
        "Méd. Desv. Hist.": "Media dos desvios historicos (previsto menos realizado) em relacao a MLT", # https://www.ons.org.br/Paginas/sobre-ons/indicadores-ons/indicador_IPE.aspx
        "Desvio ENA": "desvios em porcentagem, (ENA_previsto - ENA_realizado) / ENA_MLT",
        "IMP": "Indice de Melhoria da Previsao",
        "IMP2": "Indice de Melhoria da Previsao 2",
    }
    DATA_TYPES_DICT = {
        "Ano": "datetime64[ns]",
        "Mês": "int64",
        "Região": "object",
        "Rev. PMO": "object",
        "ENA_obs": "float64",
        "ENA_prev": "float64",
        "MLT": "float64",
        "Méd. Desv. Hist.": "float64",
        "Desvio ENA": "float64",
        "IMP": "float64",
        "IMP2": "float64",
    }
    DEFAULT_PATH = "Dados/raw/IPE"
    name = "indicador_previsao_ena"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                indicador_previsao_ena = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", na_values="---")
                for row in range(len(indicador_previsao_ena)):
                    mes = indicador_previsao_ena.iloc[row]['Mês']
                    indicador_previsao_ena.loc[row,'Ano'] = f"{indicador_previsao_ena.iloc[row]['Ano']}/{mes:02d}/01"
                data.append(indicador_previsao_ena)

        return pd.concat(data, ignore_index=True).astype(cls.DATA_TYPES_DICT)
