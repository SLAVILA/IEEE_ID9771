from abc import ABC, abstractmethod
import os


class AbstractData():
    @classmethod
    @property
    @abstractmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def DATA_DICT(cls):
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def DATA_TYPES_DICT(cls):
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def DEFAULT_FILENAME(cls):
        raise NotImplementedError

    @classmethod
    @property
    def columns(cls):
        return list(cls.DATA_DICT)

    @classmethod
    def get_data(cls, base_dir, path=None):
        if not path:
            path = cls.DEFAULT_PATH
            
        splitted = path.split("/")
        if len(splitted) <= 1:
            splitted = path.split("\\")
        
        return cls._get_data(os.path.join(base_dir, *splitted))

    @staticmethod
    @abstractmethod
    def _get_data(path):
        raise NotImplementedError