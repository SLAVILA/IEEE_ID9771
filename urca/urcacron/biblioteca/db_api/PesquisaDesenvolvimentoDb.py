import pandas as pd
from sqlalchemy import create_engine, text
import os
import time
from .abstract_db import AbstractDB


class PesquisaDesenvolvimentoDb(AbstractDB):
    """
        Esta classe faz a comunicação com o banco de dados utilizando as 
    configurações contidas no arquivo '.env' localizado na pasta 'config'.
    """

    #   Transforma esta classe em um Singleton, ou seja, só permite a 
    # existência de 1 instância desta classe
    __instance = None
    __db = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if (self.__initialized):
            return
        self.__initialized = True

        DB_SERVER = os.getenv("DB_SERVER")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_PORT = os.getenv("DB_PORT")
        DB = "pesquisa_desenvolvimento" #os.getenv("DB")
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

        self.__CONN_URCA = create_engine(
            'mssql+pyodbc://%s:%s@%s:%s/%s?driver=ODBC+Driver+17+for+SQL+Server'
            %(DB_USER, DB_PASSWORD, DB_SERVER, DB_PORT, DB)
        )
        self.__db = DB
        super().__init__()

    # retorna os nomes de todas as tabelas do banco de dados
    def __get_tables__(self):
        tables = pd.read_sql(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' AND 
            TABLE_CATALOG='%s'
            """ %(self.__db),
            self.__CONN_URCA)
        return list(tables["TABLE_NAME"])
    
    def upload_df(self, df, table):
        with self.__CONN_URCA.begin() as conn:
            df.to_sql(table, con=conn, schema='urca', if_exists='append', index=False)
            
    def command(self, command):
        with self.__CONN_URCA.begin() as conn:
            return conn.execute(text(command))

    def get_table(self, table: str, **kwargs) -> pd.DataFrame:
        if kwargs:
            if "path" not in kwargs:
                kwargs["path"] = self.proj_path
        else:
            kwargs = {"path": self.proj_path}

        sql = "SELECT * from urca.%s" %(table)
        df = pd.read_sql(sql, self.__CONN_URCA)

    # retorna as colunas de uma determinada tabela
    def get_columns(self, table: str) -> pd.DataFrame:        
        sql = """SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                 WHERE TABLE_NAME = N'%s'""" %(table)
        return pd.read_sql(sql, self.__CONN_URCA)

if __name__ == "__main__":
    print(PesquisaDesenvolvimentoDb().tables)