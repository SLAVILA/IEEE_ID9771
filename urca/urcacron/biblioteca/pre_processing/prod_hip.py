import pandas as pd
import numpy as np

class Product_hip:
    name = "Hipótese de produto"
    
    
    def implemented():
        return {
            "rolloff M+0 SE" : lambda df: Product_hip.rolloff(df, 0, "SE"),
            "rolloff M+1 SE" : lambda df: Product_hip.rolloff(df, 1, "SE"),
            "rolloff M+2 SE" : lambda df: Product_hip.rolloff(df, 2, "SE"),
            "rolloff M+3 SE" : lambda df: Product_hip.rolloff(df, 3, "SE"),
            "rolloff diferenca cumulativa M+0 SE" : lambda df: Product_hip.rolloff_diferenca_cumulativa(df, 0, "SE"),
            "rolloff diferenca cumulativa M+1 SE" : lambda df: Product_hip.rolloff_diferenca_cumulativa(df, 1, "SE"),
            "rolloff diferenca cumulativa M+2 SE" : lambda df: Product_hip.rolloff_diferenca_cumulativa(df, 2, "SE"),
            "rolloff diferenca cumulativa M+3 SE" : lambda df: Product_hip.rolloff_diferenca_cumulativa(df, 3, "SE"),
            "rolloff suavizado M+0 SE" : lambda df: Product_hip.rolloff_suavizado(df, 0, "SE"),
            "rolloff suavizado M+1 SE" : lambda df: Product_hip.rolloff_suavizado(df, 1, "SE"),
            "rolloff suavizado M+2 SE" : lambda df: Product_hip.rolloff_suavizado(df, 2, "SE"),
            "rolloff suavizado M+3 SE" : lambda df: Product_hip.rolloff_suavizado(df, 3, "SE"),
        }
    
    
    def descr():
        return {
            "rolloff M+0 SE" : "Rolloff dos podutos por maturação mensal",
            "rolloff M+1 SE" : "Rolloff dos podutos por maturação mensal",
            "rolloff M+2 SE" : "Rolloff dos podutos por maturação mensal",
            "rolloff M+3 SE" : "Rolloff dos podutos por maturação mensal",
            "rolloff diferenca cumulativa M+0 SE" : "Rolloff diferenca cumulativa dos podutos por maturação mensal",
            "rolloff diferenca cumulativa M+1 SE" : "Rolloff diferenca cumulativa dos podutos por maturação mensal",
            "rolloff diferenca cumulativa M+2 SE" : "Rolloff diferenca cumulativa dos podutos por maturação mensal",
            "rolloff diferenca cumulativa M+3 SE" : "Rolloff diferenca cumulativa dos podutos por maturação mensal",
            "rolloff suavizado M+0 SE" : "Rolloff suavizado dos podutos por maturação mensal",
            "rolloff suavizado M+1 SE" : "Rolloff suavizado dos podutos por maturação mensal",
            "rolloff suavizado M+2 SE" : "Rolloff suavizado dos podutos por maturação mensal",
            "rolloff suavizado M+3 SE" : "Rolloff suavizado dos podutos por maturação mensal",
        }
    
    
    def rolloff(df, M, submercado="SE"):
        data = df[df.M==M]
        if submercado:
            data = data[data.submercado==submercado]
        return data
    
    
    def rolloff_diferenca_cumulativa(df, M, submercado="SE", data_col="data", value_col="VWAP", product_col="produto"):
        if submercado:
            df = df[df.submercado==submercado]
        df = df.sort_values(data_col)

        # para cada produto, calcula a diferença de VWAP de um dia para o outro
        def diff(df):
            df[f"{value_col}_diff"] = df[value_col] - df[value_col].shift(1)
            return df
        data = df.groupby(product_col, group_keys=False).apply(diff)
        
        # Seleciona a maturação M
        data = data[data.M == M].sort_values(data_col).copy()
        
        # coloca o primeiro valor da série de diferenças como 0
        data.loc[data.index[0], f"{value_col}_diff"] = 0 
        
        # remove possíveis dados faltantes (primeira transação de um produto)
        data = data.loc[data[f"{value_col}_diff"].dropna().index]
        
        # reconstroí a série fazendo a soma cumulativa
        data[value_col] = data[f"{value_col}_diff"].cumsum()
        
        del data[f"{value_col}_diff"]
        return data
    
    
    def linear(day):
        days = day.daysinmonth
        return (1 - (day.day-1)/days)
    
    
    def rolloff_suavizado(df, M, submercado="SE", proportion=linear):
        if submercado:
            df = df[df.submercado==submercado]
        data = df[df.M==M].copy().sort_values("data")
        
        data_next = df[df.M>M].set_index(["data", "expiracao"])
        data_needed = data.copy()
        data_needed.expiracao += data_needed.expiracao.apply(lambda d: pd.Timedelta(days=d.daysinmonth))
        data_needed = data_needed.set_index(["data", "expiracao"])
        new_index = data_next.index.union(data_needed.index)
        data_next = data_next.reindex(new_index).sort_index(level=[1, 0])
        data_next.update(data_next.groupby("expiracao", group_keys=True).fillna(method="ffill"))
        data_next = pd.DataFrame([data_next.loc[i] for i in data_needed.index.values])
        data_next.index = data_next.index.set_names(["data", "expiracao"])
        data_next = data_next.reset_index().set_index("data")

        data["proportions"] = data.data.apply(proportion)
        data = data.reset_index().set_index("data")
        
        data_next["VWAP"] = data_next["VWAP"].fillna(data['VWAP'])
        
        data["VWAP"] = data.VWAP*data.proportions + (1-data.proportions)*data_next.VWAP
        result = data.reset_index().set_index("index")
        result.index = result.index.set_names(None)
        
        return result[df.columns]
"""
import pandas as pd
import numpy as np

class Product_hip:
    name = "Hipótese de produto"
    
    
    def implemented():
        return {
            "rolloff" : Product_hip.rolloff,
            "rolloff diferenca_cumulativa" : Product_hip.rollof_diferenca_cumulativa,
        }
    
    
    def descr():
        return {
            "rolloff" : "Rolloff dos podutos por maturação mensal",-
            "rolloff diferenca_cumulativa" : "Rolloff diferenca_cumulativa dos podutos por maturação mensal",-
        }
    
    
    def rolloff(df, M, submercado="SE"):
        data = df[df.M==M]
        if submercado:
            data = data[data.submercado==submercado]
        return data
    
    
    def rollof_diferenca_cumulativa(df, M, submercado="SE", data_col="data", value_col="VWAP", product_col="produto"):
        if submercado:
            df = df[df.submercado==submercado]
        df = df.sort_values(data_col)

        def diff(df):
            df[f"{value_col}_diff"] = df[value_col] - df[value_col].shift(1)
            return df

        data = df.groupby(product_col, group_keys=False).apply(diff)
        data = data[data.M == M]
        data[f"{value_col}_diff"].iloc[0] = data[value_col].iloc[0]
        data = data.dropna()
        data[value_col] = data[f"{value_col}_diff"].cumsum()
        del data[f"{value_col}_diff"]
        return data
"""