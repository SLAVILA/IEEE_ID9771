import pandas as pd
import os

from ..ampere import AmpereDB

from .abstract_data import AbstractData
''' Energia natural afluente histórica '''


class AmpereEnaHistRee(AbstractData):
    DATA_DICT = {
        "data": "",
        "REE": "reservatorio equivalente de energia",
        "ENA": "",
        "cod_ree": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "ENA": "float64",
        "REE": "object",
        "cod_ree": "int",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/rees"
    name = "ampere_ENA_hist_REE"

    @classmethod
    def _get_data(cls, path):
        return AmpereEnaHistRee._get_ena(path, "historica").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def _get_ena(cls, dir_path, tipo):
        rees = []
        for REE in os.listdir(dir_path):
            cod_ree = REE.split("_")[0]
            ree = REE.split("_")[1]
            path = os.path.join(dir_path, REE)
            for f in os.listdir(path):
                f_path = os.path.join(path, f)
                if os.path.isdir(f_path):
                    continue
                f_tipo = f.split("_")[-1][:-4]
                if f_tipo != tipo:
                    continue
                temp = pd.read_csv(f_path, sep=";", decimal=",", header=None).rename(columns={0: "data", 1: "ENA"})
                temp["REE"] = ree
                temp["cod_ree"] = cod_ree
                temp['data'] = pd.to_datetime(temp['data'], format="%d/%m/%Y")
                rees.append(temp.set_index("data"))

        return pd.concat(rees, axis=0).reset_index()


class AmpereEnaAtualRee(AmpereEnaHistRee):
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "ENA": "float64",
        "REE": "object",
        "cod_ree": "int",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/rees"
    name = "ampere_ENA_atual_REE"

    @classmethod
    def _get_data(cls, path):
        return super()._get_ena(path, "atual").astype(cls.DATA_TYPES_DICT)

"""Energia natural afluente histórica """

class AmpereEnaHistReeNew(AbstractData):
    DATA_DICT = {
        "data": "",
        "REE": "reservatorio equivalente de energia",
        "ENA": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        '01_sudeste': "float64",
        '02_sul': "float64",
        '03_nordeste': "float64",
        '04_norte': "float64",
        '05_itaipu': "float64",
        '06_madeira': "float64",
        '07_teles-pires': "float64",
        '08_belo-monte': "float64",
        '09_manaus-amapa': "float64",
        '10_parana': "float64",
        '11_iguacu': "float64",
        '12_paranapanema': "float64",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/diaria/02b_d_rees_ena_historica_diaria.csv"
    name = "ampere_ENA_hist_REE_new"

    @classmethod
    def _get_data(cls, path):
        df = pd.read_csv(path, sep=";", decimal=",").dropna()
        df = df.rename(columns={
            "data\sub": "data",
            "REE SUDESTE": '01_sudeste',
            "REE MADEIRA": '06_madeira',
            "REE TELES PIRES": '07_teles-pires',
            "REE ITAIPU": '05_itaipu',
            "REE PARANA": '10_parana',
            "REE PARANAPANEMA": '12_paranapanema',
            "REE SUL": '02_sul',
            "REE IGUACU": '11_iguacu',
            "REE NORDESTE": '03_nordeste',
            "REE NORTE": '04_norte',
            "REE BELO MONTE": '08_belo-monte',
            "REE MANAUS": '09_manaus-amapa'
        })
        df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")
        df = df.astype(cls.DATA_TYPES_DICT)
        df = df.set_index("data")

        new = AmpereDB().get_table("ampere_ena_hist_rees")
        new = new.set_index("data")

        ena = pd.concat([df.drop(df.index.intersection(new.index)), new])
        return ena.sort_index().reset_index()

class AmpereEnaHistReeNew2(AbstractData):
    DATA_DICT = {
        "data": "",
        "REE": "reservatorio equivalente de energia",
        "ENA": "",
        "cod_ree": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "ENA": "float64",
        "REE": "object",
        "cod_ree": "int",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/diaria/02b_d_rees_ena_historica_diaria.csv"
    name = "ampere_ENA_hist_REE_new2"

    @classmethod
    def _get_data(cls, path):
        ena = AmpereEnaHistReeNew._get_data(path)

        dfs = []
        ree = [
            "01_sudeste",
            "02_sul",
            "03_nordeste",
            "04_norte",
            "05_itaipu",
            "06_madeira",
            "07_teles-pires",
            "08_belo-monte",
            "09_manaus-amapa",
            "10_parana",
            "11_iguacu",
            "12_paranapanema"
        ]
        for r in ree:
            num, name = r.split("_")
            temp_df = ena.set_index("data")[r].dropna()
            dfs.append(pd.DataFrame({
                "data": temp_df.index, 
                "ENA": temp_df.values, 
                "REE": [name]*len(temp_df),
                "cod_ree": [int(num)]*len(temp_df)
            }).dropna())
        return pd.concat(dfs).astype(cls.DATA_TYPES_DICT)

'''
class AmpereEnaHistRee(AbstractData):
    DATA_DICT = {
        "data": "",
        "REE": "reservatorio equivalente de energia",
        "ENA": "",
        "cod_ree": "",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "ENA": "float64",
        "REE": "object",
        "cod_ree": "int",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/rees"
    name = "ampere_ENA_hist_REE"

    @classmethod
    def _get_data(cls, path):
        return AmpereEnaHistRee._get_ena(path, "historica").astype(cls.DATA_TYPES_DICT)

"""

class AmpereEnaAtualRee(AbstractData):
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "ENA": "float64",
        "REE": "object",
        "cod_ree": "int",
    }
    DEFAULT_PATH = "Dados/Ampere/item6-Energia_Natural_Afluente_histórica/rees"
    name = "ampere_ENA_atual_REE"

    @classmethod
    def _get_ena(cls, dir_path, tipo):
        rees = []
        for REE in os.listdir(dir_path):
            cod_ree = REE.split("_")[0]
            ree = REE.split("_")[1]
            path = os.path.join(dir_path, REE)
            for f in os.listdir(path):
                f_path = os.path.join(path, f)
                if os.path.isdir(f_path):
                    continue
                f_tipo = f.split("_")[-1][:-4]
                if f_tipo != tipo:
                    continue
                temp = pd.read_csv(f_path, sep=";", decimal=",", header=None).rename(columns={0: "data", 1: "ENA"})
                temp["REE"] = ree
                temp["cod_ree"] = cod_ree
                temp['data'] = pd.to_datetime(temp['data'], format="%d/%m/%Y")
                rees.append(temp.set_index("data"))

        return pd.concat(rees, axis=0).reset_index()
    
    @classmethod
    def _get_data(cls, path):
        return AmpereEnaAtualRee._get_ena(path, "atual").astype(cls.DATA_TYPES_DICT)

'''