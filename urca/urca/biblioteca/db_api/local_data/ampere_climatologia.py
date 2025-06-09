import pandas as pd
import os

from .abstract_data import AbstractData
''' Climatologia de precipitação, vazão natural afluente e energia natural afluente '''


class AmpereClimatologiaEnaBacia(AbstractData):
    DATA_DICT = {
        "bacia\mes": "coluna referenta as bacias",
        "1": "mes de janeiro",
        "2": "fev",
        "3": "mar",
        "4": "abr",
        "5": "mai",
        "6": "jun",
        "7": "jul",
        "8": "ago",
        "9": "set",
        "10": "out",
        "11": "nov",
        "12": "dez",
        
    }
    DATA_TYPES_DICT = {
        "bacia\mes": "object",
        "1": "float64",
        "2": "float64",
        "3": "float64",
        "4": "float64",
        "5": "float64",
        "6": "float64",
        "7": "float64",
        "8": "float64",
        "9": "float64",
        "10": "float64",
        "11": "float64",
        "12": "float64",
        
    }
    DEFAULT_PATH = "Dados/Ampere/item5-caracterizacao-climatologica"
    name = "ampere_climatologia_ena_bacia"

    @classmethod
    def _get_data(cls, path):
        return AmpereClimatologiaEnaBacia._get_climatologia(path, "bacias", "ena").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _get_climatologia(cls, dir_path, variavel, tipo):
        data = []
        for arquivo in os.listdir(dir_path):
            f_variavel = arquivo.split("_")[0]
            if f_variavel != variavel:
                continue
            f_tipo = arquivo.split("_")[-1][:-4]
            if f_tipo != tipo:
                continue
            #print(f_tipo)
            df = pd.read_csv(os.path.join(dir_path, arquivo), sep=";", decimal=",", header=0, encoding='Windows-1252')
            data.append(df)
        return pd.concat(data, axis=0)#.reset_index()


class AmpereClimatologiaPrecipitacaoBacia(AmpereClimatologiaEnaBacia):
    DATA_DICT = {
        # mesmo que o de AmpereClimatologiaEnaBacia
    }
    DATA_TYPES_DICT = {
        "bacia\mes": "object",
        "1": "float64",
        "2": "float64",
        "3": "float64",
        "4": "float64",
        "5": "float64",
        "6": "float64",
        "7": "float64",
        "8": "float64",
        "9": "float64",
        "10": "float64",
        "11": "float64",
        "12": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item5-caracterizacao-climatologica"
    name = "ampere_climatologia_precipitacao_bacia"

    @classmethod
    def _get_data(cls, path):
        return super()._get_climatologia(path, "bacias", "precipitação").astype(cls.DATA_TYPES_DICT)


class AmpereClimatologiaPrecipitacaoREE(AmpereClimatologiaEnaBacia):
    DATA_DICT = {
        "ree\mes": "coluna referenta aos rees",
        "1": "mes de janeiro",
        "2": "fev",
        "3": "mar",
        "4": "abr",
        "5": "mai",
        "6": "jun",
        "7": "jul",
        "8": "ago",
        "9": "set",
        "10": "out",
        "11": "nov",
        "12": "dez",
    }
    DATA_TYPES_DICT = {
        "ree\mes": "object",
        "1": "float64",
        "2": "float64",
        "3": "float64",
        "4": "float64",
        "5": "float64",
        "6": "float64",
        "7": "float64",
        "8": "float64",
        "9": "float64",
        "10": "float64",
        "11": "float64",
        "12": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item5-caracterizacao-climatologica"
    name = "ampere_climatologia_precipitacao_ree"

    @classmethod
    def _get_data(cls, path):
        return super()._get_climatologia(path, "rees", "precipitação").astype(cls.DATA_TYPES_DICT)


class AmpereClimatologiaEnaREE(AmpereClimatologiaEnaBacia):
    DATA_DICT = {
        # mesmo que o de AmpereClimatologiaPrecipitacaoREE
    }
    DATA_TYPES_DICT = {
        "ree\mes": "object",
        "1": "float64",
        "2": "float64",
        "3": "float64",
        "4": "float64",
        "5": "float64",
        "6": "float64",
        "7": "float64",
        "8": "float64",
        "9": "float64",
        "10": "float64",
        "11": "float64",
        "12": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item5-caracterizacao-climatologica"
    name = "ampere_climatologia_ena_ree"

    @classmethod
    def _get_data(cls, path):
        return super()._get_climatologia(path, "rees", "ena").astype(cls.DATA_TYPES_DICT)


class AmpereClimatologiaEnaSUBs(AmpereClimatologiaEnaBacia):
    DATA_DICT = {
        # mesmo que o de AmpereClimatologiaPrecipitacaoREE
    }
    DATA_TYPES_DICT = {
        "ree\mes": "object",
        "1": "float64",
        "2": "float64",
        "3": "float64",
        "4": "float64",
        "5": "float64",
        "6": "float64",
        "7": "float64",
        "8": "float64",
        "9": "float64",
        "10": "float64",
        "11": "float64",
        "12": "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item5-caracterizacao-climatologica"
    name = "ampere_climatologia_ena_subs"

    @classmethod
    def _get_data(cls, path):
        return super()._get_climatologia(path, "subs", "ena").astype(cls.DATA_TYPES_DICT)