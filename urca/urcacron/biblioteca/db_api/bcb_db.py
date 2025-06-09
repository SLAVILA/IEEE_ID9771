from .abstract_db import AbstractDB
import pandas as pd
import requests
import time
from bcb import Expectativas


class BcbDb(AbstractDB):
    """
        Esta classe faz a comunicação a api do Banco central do Brasil
    """

    __cod2name = {
        189:   "Índice geral de preços do mercado (IGP-M)",
        433:   "Índice nacional de preços ao consumidor - amplo (IPCA)",
        1207:  "Produto interno bruto em R$ correntes",
        1208:  "Produto interno bruto em R$ do último ano",
        1431:  "Consumo de energia elétrica - Região Sudeste - Total",
        4192:  "PIB acumulado dos últimos 12 meses - Em US$ milhões",
        4380:  "PIB mensal - Valores correntes (R$ milhões)",
        4381:  "PIB acumulado no ano - Valores correntes (R$ milhões)",
        4382:  "PIB acumulado dos últimos 12 meses - Valores correntes (R$ milhões)",
        22084: "PIB trimestral - Dados observados - Indústria (total)",
        24363: "Índice de Atividade Econômica do Banco Central - IBC-Br",
        27577: "Índice de Commodities - Brasil - Energia",
        27863: "Índice Nacional de Preços ao Consumidor - Amplo (IPCA) - Industriais"
    }
    __name2cod = {}

    def __init__(self):
        self.__name2cod = {nome: cod for cod, nome in self.__cod2name.items()}
        super().__init__()

    # retorna os nomes de todas as tabelas do banco de dados
    def __get_tables__(self):
        return list(self.__name2cod)

    def get_code(self, table: str):
        return self.__name2cod[table]

    def get_table(self, table: str, max_tries=10, sleep_time=0, verbose=False) -> pd.DataFrame:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.%i/dados?formato=json" %self.get_code(table)
        for i in range(1, max_tries+1):
            try:
                response = requests.get(url)
                df = pd.DataFrame(response.json())
                return df
            except requests.exceptions.JSONDecodeError as e:
                if i == max_tries:
                    raise e

                if verbose:
                    print("   Erro, tentando novamente (%i/%i)" %(i, max_tries))
                if sleep_time>0:
                    time.sleep(sleep_time)

    # retorna as colunas de uma determinada tabela
    def get_columns(self, table: str) -> pd.DataFrame:
        return self.get_table(table).columns


class ExpectativasBcb(AbstractDB):
    def __init__(self):
        self.__name2cod = {nome: cod for cod, nome in self.__cod2name.items()}
        super().__init__()
    def describe(exp):
        em = Expectativas()
        return em.describe(exp)
    def get_table(mercado, indicador, base_calculo=1):
        em = Expectativas()
        ep = em.get_endpoint(mercado)
        dados_expec = (ep.query()
                       .filter(ep.Indicador == indicador)
                       .filter(ep.baseCalculo == base_calculo)
                       .filter(ep.Data >= "2000-01-01")
                       .select(ep.Indicador, ep.Data, ep.Mediana, ep.DataReferencia)
                       .collect()
                       )
        return dados_expec
    def get_df(mercado, indicador, base_calculo=1):
        df = ExpectativasBcb.get_table(mercado, indicador, base_calculo)
        df['DataReferencia'] = pd.to_datetime(df['DataReferencia'], format='%m/%Y')
        df = df.sort_values(by=["Data", "DataReferencia"])
        df['num'] = df.groupby('Data').cumcount()
        df_pivot = df.pivot(index='Data', columns='num', values='Mediana')
        df_pivot.columns = ['M+' + str(col) for col in df_pivot.columns]
        df_pivot = df_pivot.assign(**{f'M+{i}': [valor for valor in df_pivot['M+24']] for i in range(24, 36+1)})
        df_pivot = df_pivot.ffill(axis=1)
        df_pivot = df_pivot.reset_index()
        return (df_pivot)