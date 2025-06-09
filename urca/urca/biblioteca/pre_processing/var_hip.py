import pandas as pd
import numpy as np

class Variable_hip:
    name = "Hipótese de variável"
    
    @staticmethod
    def implemented():
        return {
            "hip_2" : lambda df: Variable_hip.hip_2(df, output_col="result"),
            "hip_3" : lambda df: Variable_hip.hip_3(df, output_col="result"),
            "hip_12": lambda df: Variable_hip.hip_12(df, output_col="result"),
            "hip_15": lambda df: Variable_hip.hip_15(df, output_col="result"),
        }
    
    @staticmethod
    def descr():
        return {
            "hip_2" : "Hipótese 2 - a elasticidade dinâmica do preço e do volume, usando a média aritmética do s(VWAP) e q(acumulado) para os dias h e h-1",
            "hip_3" : "Hipótese 3 - a elasticidade dinâmica do preço, usando o s(VWAP) para os dias h e h-1 - return = elasticidade",
            "hip_12": "Hipótese 12 - o retorno pelo log-neperiano ",
            "hip_15": "Hipótese 15 - o retorno pelo log-neperiano normalizado pela maturidade total",
        }
    
    @staticmethod
    def hip_2(df, price_col="VWAP", vol_col="volume", output_col='hip_2_elasticidade_q/s', data_col='data', clip=True, clip_lower=-3.0, clip_upper=3.0, abs_val=True):
        ''' 
        Hipótese 2 - a elasticidade dinâmica do preço e do volume, usando a média aritmética do s(VWAP) e q(acumulado) para os dias h e h-1
        '''
        # Ordena por data
        df = df.sort_values(data_col)
        
        # Valores passados
        price_past = df[price_col].shift(1)
        vol_past = df[vol_col].shift(1)
    
        # variação
        price_delta = df[price_col]-price_past
        vol_delta = df[vol_col]-vol_past
        
        # média
        price_mean = (df[price_col]+price_past)/2
        vol_mean = (df[vol_col]+vol_past)/2
        
        df_dict = {
            output_col: vol_delta/price_delta * price_mean/vol_mean
        }
        res_df = pd.concat([df, pd.DataFrame(df_dict)], axis=1)
        
        if clip:
            res_df[output_col] = res_df[output_col].clip(clip_lower, clip_upper)
        if abs_val:
            res_df[output_col] = res_df[output_col].abs()
        return res_df

    @staticmethod
    def hip_3(df, price_col="VWAP", output_col="hip_3_retorno", data_col='data'):
        ''' 
        Hipótese 3 - a elasticidade dinâmica do preço, usando o s(VWAP) para os dias h e h-1 - return = elasticidade
        '''
        # Ordena por data
        df = df.sort_values(data_col)
        
        #transforma em retornos 
        df_dict = {
            output_col: df[price_col]/(df[price_col].shift(1)) - 1
        }
        res_df = pd.concat([df, pd.DataFrame(df_dict)], axis=1)
        return res_df

    @staticmethod
    def hip_12(df, price_col="VWAP", output_col="hip_12_retorno_log_np", data_col='data'):
        '''
        Hipótese 12 - o retorno pelo log-neperiano 
        '''
        # Ordena por data
        df = df.sort_values(data_col)
        
        # transforma os preços em retornos expressos por logaritmo neperiano (natural)
        # np.log() converte para log np e após a função .diff() calcula os retornos
        df_dict = {
            output_col: np.log(df[price_col]).diff()
        }
        res_df = pd.concat([df, pd.DataFrame(df_dict)], axis=1)
        return res_df

    @staticmethod
    def hip_15(df, price_col="VWAP", mat_col="H", output_col="hip_15_retorno_log_np_norm", data_col='data'):
        ''' 
        Hipótese 15 - o retorno pelo log-neperiano normalizado pela maturidade total
        '''
        # Ordena por data
        df = df.sort_values(data_col)
        
        # transforma os preços VWAP em logaritmo neperiano
        # np.log é padrão para o log natural aka log neperiano
        df_dict = {
            output_col: np.log(df[price_col])-np.log(df[mat_col])
        }
        res_df = pd.concat([df, pd.DataFrame(df_dict)], axis=1)
        return res_df
