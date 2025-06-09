import pandas as pd
from sqlalchemy import create_engine, text
import os
import time

from ee_ampere_consultoria.produtos.flux import FluxAutomatico

from .abstract_db import AbstractDB
from .ampere_data import EnaPrevCFSV2_REE, EnaHist_REE


class AmpereDB(AbstractDB):
    """
        Esta classe faz a comunicação com o banco de dados da Ampere utilizando as 
    configurações contidas no arquivo '.env' localizado na pasta 'config'.
    """

    #   Transforma esta classe em um Singleton, ou seja, só permite a 
    # existência de 1 instância desta classe
    __instance = None

    # Tabela criada a partir de outra tabela
    __TABLES = {
        EnaPrevCFSV2_REE.name: EnaPrevCFSV2_REE,
        EnaHist_REE.name: EnaHist_REE
    }

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def get_flux(self):
        USERNAME = os.getenv("AMPERE_USER")
        MD5_PASSWORD_HASH = os.getenv("AMPERE_PASSWORD")
        USER_ACESS_TOKEN = os.getenv("AMPERE_TOKEN")
        try:
            assert USERNAME is not None
            assert MD5_PASSWORD_HASH is not None
            assert USER_ACESS_TOKEN is not None
        except AssertionError:
            raise AssertionError(
                "Não foi possível carregar variáveis do ambiente. " +
                "Verifique se possuí o arquivo '.env' na pasta 'config'.")
        
        return FluxAutomatico(USERNAME, MD5_PASSWORD_HASH, USER_ACESS_TOKEN)
    
    def __init__(self):
        if (self.__initialized):
            return
        test = self.get_flux()
        self.__initialized = True
        super().__init__()

    def __get_table_class(self, table):
        try:
            return self.__TABLES[table]
        except KeyError:
            raise KeyError(f'"{table}" not found in Local tables. See Local().tables')

    # retorna os nomes de todas as tabelas do banco de dados
    def __get_tables__(self):
        return list(self.__TABLES)

    def get_table(self, table: str, update=False) -> pd.DataFrame:
        return self.__get_table_class(table).get_data(
            self.proj_path,
            self.get_flux(),
            update=update
        )

    # retorna as colunas de uma determinada tabela
    def get_columns(self, table: str) -> pd.DataFrame:
        return self.__get_table_class(table).columns

    # retorna as colunas de uma determinada tabela
    def get_data_dict(self, table: str) -> pd.DataFrame:
        return self.__get_table_class(table).DATA_DICT

    # retorna as colunas de uma determinada tabela
    def get_type_dict(self, table: str) -> pd.DataFrame:
        return self.__get_table_class(table).DATA_TYPES_DICT