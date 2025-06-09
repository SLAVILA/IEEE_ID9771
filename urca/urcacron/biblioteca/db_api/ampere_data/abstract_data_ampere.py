from abc import abstractmethod
import os
from ..local_data.abstract_data import AbstractData


class AbstractDataAmpere(AbstractData):
    @classmethod
    def get_data(cls, base_dir, flux=None, update=False):
        path = cls.DEFAULT_PATH
            
        splitted = path.split("/")
        if len(splitted) <= 1:
            splitted = path.split("\\")
        
        if update:
            if flux is None:
                raise ValueError("FluxAutomatico não foi fornecido para realizar atualização dos dados")
            cls._update_data(flux, os.path.join(base_dir, *splitted))
        return cls._get_data(os.path.join(base_dir, *splitted))

    @staticmethod
    @abstractmethod
    def _get_data(flux, path):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def _update_data(flux, path):
        raise NotImplementedError