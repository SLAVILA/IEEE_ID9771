from biblioteca.db_api.urca_data.pre_processamento import preco_bbce_men_preco_fixo_diario
from biblioteca.db_api.urca_data.utils import get_maturidade

from matplotlib import pyplot as plt
import pandas as pd

    
class PricesFillMissingFunctions():
    @staticmethod
    def implemented():
        return {
            "VWAP pentada": PricesFillMissingFunctions.VWAP_grouped,
            "linear": PricesFillMissingFunctions.linear
        }
    
    @staticmethod
    def VWAP_grouped(df, fill_func=None):
        if fill_func:
            df = fill_func(df)
        df["VWAP"] = (df["volume"]*df["VWAP"]).rolling(5, min_periods=1).sum()
        df["volume"] = df["volume"].rolling(5, min_periods=1).sum()
        df["VWAP"] = (df["VWAP"]/df["volume"]).ffill()
        df["volume"] = df["volume"].fillna(0)
        return df

    @staticmethod
    def linear(df, fill_func=None):
        if fill_func:
            df = fill_func(df)
        df["VWAP"] = df["VWAP"].interpolate()
        return df
    
    """
    def lin_interpol(a, fill_func=None):
        if fill_func:
            a = fill_func(a)
        a["VWAP"] = (a["volume"]*a["VWAP"]).rolling(5, min_periods=1).sum()
        a["volume"] = a["volume"].rolling(5, min_periods=1).sum()
        a["VWAP"] = (a["VWAP"]/a["volume"]).ffill()
        a["volume"] = a["volume"].fillna(0)
        return a
    """


class PricesFillMissing():
    @staticmethod
    def fill_dates_poduct(ffill_cols=["produto", "submercado", "expiracao"], lst_date=None):
        # preenche com None datas até lst_date ou a expiração do produto
        def temp(a, ffill_cols, lst_date):
            if lst_date is not None:
                lst_date = min(lst_date, a.expiracao.iloc[0]-pd.Timedelta(days=1))
            else:
                lst_date = a.data.max()
            idx = pd.Index(pd.date_range(a.data.min(), lst_date), name='data')
            a = a.set_index("data").reindex(idx, fill_value=None)
            a[ffill_cols] = a[ffill_cols].ffill()
            return a.reset_index()
        return lambda a: temp(a, ffill_cols, lst_date)

    @staticmethod
    def interpolate_prod(preco, func):
        # aplica a interpolação sobre cada produto individualmente
        lst_date = preco.data.max()
        grouped = preco.groupby(["produto", "submercado", "expiracao"], group_keys=False)
        preco = grouped.apply(func, PricesFillMissing.fill_dates_poduct(lst_date=lst_date))
        return get_maturidade(preco, "produto", "data").reset_index(drop=True)

    @staticmethod
    def filter_precos(preco_raw, submercado="SE"):
        # realiza a filtragem inicial dos dados
        preco = preco_raw[preco_raw.submercado == submercado]
        preco = preco[preco.expiracao > "01/01/2015"].sort_values(["expiracao", "data"])
        preco = preco[preco.data < preco.expiracao]
        return preco.reset_index(drop=True)

    @staticmethod
    def get_all(preco_raw):
        preco_clean = PricesFillMissing.filter_precos(preco_raw)
        processed = {
            "raw": preco_clean
        }
        for name, func in PricesFillMissingFunctions.implemented().items():
            processed[name] = PricesFillMissing.interpolate_prod(preco_clean, func)
        return processed
