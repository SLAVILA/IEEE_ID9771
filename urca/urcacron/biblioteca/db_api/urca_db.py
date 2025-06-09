import pandas as pd
import pymssql
from sqlalchemy import create_engine, text
import os
import time
from .abstract_db import AbstractDB
from .urca_data.pre_processamento import preco_bbce, preco_bbce_men_preco_fixo, preco_bbce_men_preco_fixo_diario
from time import sleep


def try_again(function):
    def decorate(self, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            sleep(1)
            self.__initialize()
            return function(*args, **kwargs)
    return function


class UrcaDb(AbstractDB):
    """
        Esta classe faz a comunicação com o banco de dados utilizando as 
    configurações contidas no arquivo '.env' localizado na pasta 'config'.
    """

    #   Transforma esta classe em um Singleton, ou seja, só permite a
    # existência de 1 instância desta classe
    __instance = None
    __db = None

    # Tabela criada a partir de outra tabela
    __new_tables = {
        "preco_bbce_men_preco_fixo": {
            "func": 
                lambda x, **kwargs: 
                    preco_bbce_men_preco_fixo(
                        x, 
                        outliers=os.path.join("Dados", "bbce_outliers", "outliers (09-10-2023).csv"), 
                        **kwargs), 
            "table": "preco_bbce" },
        "preco_bbce_men_preco_fixo_raw": {
            "func": 
                lambda x, **kwargs: preco_bbce_men_preco_fixo(x, outliers=None, **kwargs), 
            "table": "preco_bbce" },
        "preco_bbce_men_preco_fixo_diario": {
            "func": preco_bbce_men_preco_fixo_diario,
            "table": "preco_bbce_men_preco_fixo"
        }
    }
    # Tabela com função de pré-processamento
    __processed_tables = {
        "preco_bbce": preco_bbce,
    }

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if (not self.__initialized):
            self.__initialize()
        super().__init__()

    def __initialize(self):
        self.__initialized = True

        DB_SERVER = os.getenv("DB_SERVER")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_PORT = os.getenv("DB_PORT")
        DB = os.getenv("DB")
        try:
            assert DB_SERVER is not None
            assert DB_USER is not None
            assert DB_PASSWORD is not None
            assert DB_PORT is not None
            assert DB is not None
        except AssertionError:
            raise AssertionError(
                "Não foi possível carregar variáveis do ambiente. " +
                "Verifique se possuí o arquivo '.env' na pasta 'config'.")

        self.__CONN_URCA = pymssql.connect(DB_SERVER, DB_USER, DB_PASSWORD, DB)
        self.__db = DB

    # retorna os nomes de todas as tabelas do banco de dados
    @try_again
    def __get_tables__(self):
        tables = pd.read_sql(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' AND 
            TABLE_CATALOG='%s'
            """ %(self.__db),
            self.__CONN_URCA)
        return list(tables["TABLE_NAME"]) + list(self.__new_tables.keys())

    @try_again
    def get_table(self, table: str, **kwargs) -> pd.DataFrame:
        if kwargs:
            if "path" not in kwargs:
                kwargs["path"] = self.proj_path
        else:
            kwargs = {"path": self.proj_path}

        if table in self.__new_tables:
            # Tabela criada a partir de outra tabela
            df = self.get_table(self.__new_tables[table]["table"])
            return self.__new_tables[table]["func"](df, **kwargs)

        sql = "SELECT * from urca.%s" %(table)
        df = pd.read_sql(sql, self.__CONN_URCA)
        if table in self.__processed_tables:
            df = self.__processed_tables[table](df, **kwargs)
        return df

    # retorna as colunas de uma determinada tabela
    @try_again
    def get_columns(self, table: str) -> pd.DataFrame:
        if table in self.__new_tables:
            return []

        sql = """SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                 WHERE TABLE_NAME = N'%s'""" %(table)
        return pd.read_sql(sql, self.__CONN_URCA)
