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
        27863: "Índice Nacional de Preços ao Consumidor - Amplo (IPCA) - Industriais",
        -1: "expectativa_ipca"
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
        if table == "expectativa_ipca":
            return self.generate_expectativa_ipca()
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

    def generate_expectativa_ipca(self):
        def months_diff(row):
            m = row.Data.month
            y = row.Data.year
            ref_m, ref_y = row.DataReferencia.split("/")
            return int(ref_m)-m + (int(ref_y)-y)*12
        
        em = Expectativas()
        ep = em.get_endpoint("ExpectativaMercadoMensais")
        q = ep.query()
        q = q.filter(ep.Indicador == 'IPCA')
        q = q.filter(ep.baseCalculo == 1)
        q = q.select(ep.Data, ep.DataReferencia, ep.Mediana)
        a = q.collect()
        
        IPCA_exp = a
        IPCA_exp.Data = pd.to_datetime(IPCA_exp.Data, format="%Y-%m-%d")
        IPCA_exp["Month"] = IPCA_exp.apply(months_diff, axis=1)
        IPCA_exp = IPCA_exp[IPCA_exp["Month"] >= 0]
        IPCA_exp
        
        def process(df):
            data = pd.DataFrame(index=range(0, 50), columns=["Mediana"])
            data.update(df.set_index("Month"))
            data.Mediana = data.Mediana.ffill()
            dataT = data.T
            dataT.index = [df.Data.iloc[0]]
            return dataT
            
        b = IPCA_exp.groupby("Data").apply(process)
        b = b.droplevel(level=1)
        b.columns = [f"M+{i}" for i in b.columns]
        return b