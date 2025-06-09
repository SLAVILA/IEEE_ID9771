import pandas as pd
import os

from .abstract_data import AbstractData
''' Energia armazenada diária subsistemas '''


class AmpereEARdSubs(AbstractData):
    DATA_DICT = {
        "data": "coluna referenta as datas",
    }
    DATA_TYPES_DICT = {
        "data": "object",
    }
    DEFAULT_PATH = "Dados/Ampere/item4-energia_armazenada_historica"
    name = "ampere_EAR_d_subs"

    @classmethod
    def _get_data(cls, path):
        return AmpereEARdSubs._get_EARd(path, "subs", "EAR")#.astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _get_EARd(cls, dir_path, variavel, tipo):
        data = []
        for arquivo in os.listdir(dir_path):
            f_variavel = arquivo.split("_")[0]
            if f_variavel != variavel:
                continue
            f_tipo = arquivo.split("_")[-3]
            if f_tipo != tipo:
                continue
            print(f_tipo)
            df = pd.read_csv(os.path.join(dir_path, arquivo), sep=";", decimal=",", header=0)
            df = df.rename(columns={list(df.columns)[0]: 'data'})
            df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")
            data.append(df)
            data = df
        return data#pd.concat(data, axis=0)#.reset_index()