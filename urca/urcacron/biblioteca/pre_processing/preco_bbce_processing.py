from pre_processing.prices import PricesFillMissing, PricesFillMissingFunctions
from db_api import UrcaDb


class PrecoBBCE():
    def __init__(self, dados=None):
        if dados is None:
            dados = UrcaDb().get_table("preco_bbce_men_preco_fixo_diario")
        self.__orig = dados.copy()
        self.__filled = {}
        self.__preco_clean = None
        
    @property
    def raw(self):
        return self.__orig.copy()
    
    @property
    def preco_clean(self):
        if self.__preco_clean is None:
            self.__preco_clean = PricesFillMissing.filter_precos(self.__orig)
        return self.__preco_clean
        
    @property
    def fillna_methods(self):
        return list(PricesFillMissingFunctions.implemented().keys())
        
    def fillna(self, method):
        # verifica se é um método válido
        if method not in self.fillna_methods:
            raise ValueError(f"'{method}' isn't implemented")
            
        # se já foi calculado
        if method in self.__filled:
            return self.__filled[method]
        
        func = PricesFillMissingFunctions.implemented()[method]
        self.__filled[method] = PricesFillMissing.interpolate_prod(self.__preco_clean.copy(), func)
        return self.__filled[method]

    
