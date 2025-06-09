from abc import ABC, abstractmethod
import pandas as pd
import os

class AbstractDB(ABC):
    """
        Esta classe define a interface das classes desse módulo.
    """

    def __init__(self):
        self.__proj_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        self.__tables: list = self.__get_tables__()
        self.__tables.sort()

    # retorna os nomes de todas as tabelas do banco de dados
    @abstractmethod
    def __get_tables__(self) -> list:
        pass

    @property
    def tables(self) -> list:
        return self.__tables
    
    @property
    def proj_path(self) -> list:
        return self.__proj_path

    @abstractmethod
    def get_table(self, table: str) -> pd.DataFrame:
        pass

    # retorna as colunas de uma determinada tabela
    @abstractmethod
    def get_columns(self, table: str) -> pd.DataFrame:
        pass
