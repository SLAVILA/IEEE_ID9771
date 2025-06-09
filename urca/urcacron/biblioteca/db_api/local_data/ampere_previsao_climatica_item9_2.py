import pandas as pd
import os

from .abstract_data import AbstractData
from .ampere_previsao_climatica import AmperePrevClimaticaREE, AmperePrevClimaticaBacia


class AmperePrevClimaticaItem3e9_4BaciaUniao(AbstractData):
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = ""
    name = "ampere_previsao_climatica_bacia_uniao"

    @classmethod
    def _get_data(cls, path):
        bacia_complemento = AmperePrevClimaticaComplementoBacia.get_data(path)
        bacia = AmperePrevClimaticaBacia.get_data(path)
        data = pd.concat([bacia_complemento, bacia]).drop(columns="M+0").drop_duplicates()
        data = data.sort_values(by=['data'])
        return data


class AmperePrevClimaticaItem3e9_4REESUniao(AbstractData):
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = ""
    name = "ampere_previsao_climatica_rees_uniao"

    @classmethod
    def _get_data(cls, path):
        rees_complemento = AmperePrevClimaticaComplementoREE.get_data(path)
        rees = AmperePrevClimaticaREE.get_data(path)
        data = pd.concat([rees_complemento, rees]).drop(columns="M+0").drop_duplicates()
        data = data.sort_values(by=['data'])
        return data


class AmperePrevClimaticaComplementoREE(AbstractData):
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.2-previsao_climatica"
    name = "ampere_previsao_climatica_rees_complemento"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if (modelo != "_mensal" and modelo != "CPC_1981-1999_MERGE_2000-2023"):
                    path_temp = os.path.join(path, modelo, "rees")
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    
                    for df, cod, name in raw:
                        df["cod_rees"] = cod[len("ree"):]
                        df["modelo"] = modelo
                        df["nome_rees"] = name
                        data.append(df)
                    #print(data)       
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _load_data_common(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                n_col = 12
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None, names=range(n_col))
                df = df.rename({i: f"M+{i-1}" for i in range (1, len(df.columns))}, axis='columns')
                df = df.rename({0: 'data'}, axis='columns') 
                try:
                    df['data'] = pd.to_datetime(df['data'], format="%Y-%m-%d")
                except:
                    df['data'] = pd.to_datetime(df['data'], format="%Y%m")
                cod, name, _ = filename.split("_", 2)
                data.append((df, cod, name))
        return data


# +

class AmperePrevClimaticaComplementoBacia(AmperePrevClimaticaComplementoREE):
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.2-previsao_climatica"
    name = "ampere_previsao_climatica_bacias_complemento"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if (modelo != "_mensal" and modelo != "CPC_1981-1999_MERGE_2000-2023"):
                    path_temp = os.path.join(path, modelo, "bacias")
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    for df, cod, name in raw:
                        df["cod_bacia"] = cod[len("bacia"):]
                        df["modelo"] = modelo
                        df["nome_bacia"] = name
                        data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)


# +

class AmperePrevClimaticaComplementoCPCBacia(AbstractData):
    '''
    Complementação - anomalia de precipitação observada mensal
    Para que o desempenho das previsões sazonais retrospectivas e 
    operacionais possa ser avaliado, produziu-se as séries temporais 
    de anomalia de precipitação observadas em base mensal. 
    O produto entregue é a composição de dados provenientes de duas fontes distintas. 
    '''
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.2-previsao_climatica"
    name = "ampere_previsao_climatica_CPC_complemento_bacia"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if modelo == "CPC_1981-1999_MERGE_2000-2023":
                    path_temp = os.path.join(path, modelo, "bacias")
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    for df, cod, name in raw:
                        df["cod_bacia"] = cod[len("bacia"):]
                        df["modelo"] = modelo
                        df["nome_bacia"] = name
                        data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _load_data_common(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None)
                df = df.rename({0: 'data'}, axis='columns') 
                try:
                    df['data'] = pd.to_datetime(df['data'], format="%Y-%m-%d")
                except:
                    df['data'] = pd.to_datetime(df['data'], format="%Y%m")
                cod, name, _ = filename.split("_", 2)
                data.append((df, cod, name))
        return data


# +

class AmperePrevClimaticaComplementoCPCREE(AmperePrevClimaticaComplementoCPCBacia):
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.2-previsao_climatica"
    name = "ampere_previsao_climatica_CPC_complemento_ree"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if modelo == "CPC_1981-1999_MERGE_2000-2023":
                    path_temp = os.path.join(path, modelo, "rees")
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    for df, cod, name in raw:
                        df["cod_rees"] = cod[len("ree"):]
                        df["modelo"] = modelo
                        df["nome_rees"] = name
                        data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)


# +

class AmperePrevClimaticaHistMensalENA(AbstractData):
    '''
    Entende-se que análises estatísticas com base nos produtos associados aos
    subitens 3, 5 e 9 devem ser realizadas prioritariamente em base mensal,
    acompanhando a discretização temporal destes dados. A fim de complementar
    possíveis análises neste âmbito, produziu-se toda a série histórica de
    Energia Natural Afluente em base mensal para o período entre janeiro de 1931
    e dezembro de 2023 com base na configuração hidráulica do parque hidroelétrico atual.
    Os resultados foram agrupados nas bacias e reservatórios equivalentes de energia pré-determinados.
    '''
    DATA_DICT = {
        "data": "data",
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
    }
    DEFAULT_PATH = "Dados/Ampere/item9.2-previsao_climatica"
    name = "ampere_previsao_climatica_ENA_hist_mensal"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if os.path.isdir(os.path.join(path, modelo)):
                if modelo == "_mensal":
                    path_temp = os.path.join(path, modelo)
                    #print(path_temp)
                    raw = cls._load_data_common(path_temp)
                    for df, cod, name in raw:
                        df["cod"] = cod
                        df["pasta"] = modelo
                        df["nome"] = name
                        data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _load_data_common(cls, path):
        data = []
        Meses_dict = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None, encoding='windows-1252', skiprows=[0])
                df = df.rename({0: 'data'}, axis='columns')
                datas = []
                for row in range(len(df)):
                    datas.append(df['data'][row].split('/'))
                datas_ok = []
                for mes, ano in datas:
                    datas_ok.append(str(ano)+"/"+str(Meses_dict[mes]))
                df['data'] = pd.to_datetime(datas_ok, format="%y/%m")
                aaa = filename.split("_")
                cod = aaa[-2]
                name = aaa[-1]
                data.append((df, cod, name))
        return data
