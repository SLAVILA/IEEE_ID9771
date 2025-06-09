import pandas as pd
import os
from .abstract_data import AbstractData


class PojecaoCarga(AbstractData):
    DATA_DICT = {  # https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/revisoes-quadrimestrais-da-carga
        "Data_Estudo": "Primeiro dia do período de estudo",
        "Data_Estudo_EPE": "data em que o estudo foi feito",  # usar dia 01/01 do ano
        "Submercado": "nome do subsistema",
        "Data_referencia": "Primeiro dia da carga prevista",
        "Projecao_carga": "Carga média em MW",
        "Media_proj_carga": "Média do ano",  # pode ser uma string
    }
    DATA_TYPES_DICT = {
        "Data_Estudo": "datetime64[ns]", 
        "Data_Estudo_EPE" : "object", 
        "Submercado" : "object", 
        "Data_referencia" : "datetime64[ns]", 
        "Projecao_carga": "int64", 
        "Media_proj_carga": "int64"
    }
    DEFAULT_PATH = "Dados/raw/projecao_carga"
    name = "projecao_carga"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                projecao_carga = pd.read_csv(os.path.join(path, filename), sep=",")
                projecao_carga["Data_Estudo"] = projecao_carga.apply(lambda row: row.Data_Estudo_EPE[0:4]+"/01/01", axis = 1)
    
                Data_dict = { "Data_Estudo": [], "Data_Estudo_EPE" : [], "Submercado" : [], "Data_referencia" : [], "Projecao_carga":[], "Media_proj_carga":[]}
                Meses_dict = { 1: "Jan", 2: "Fev" , 3: "Mar" , 4: "Abr" , 5: "Mai" , 6: "Jun", 7:"Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
                for row in range(len(projecao_carga)):
                    for mes in range(1,13):
                        Data_dict["Data_Estudo"].append(projecao_carga["Data_Estudo"][row])
                        Data_dict["Data_Estudo_EPE"].append(projecao_carga["Data_Estudo_EPE"][row])
                        Data_dict["Submercado"].append(projecao_carga["Submercado"][row])
                        Data_dict["Data_referencia"].append(f"{projecao_carga.loc[row]['Ano_referencia']}/{mes:02d}/01")
                        Data_dict["Projecao_carga"].append(projecao_carga[Meses_dict[mes]][row])
                        Data_dict["Media_proj_carga"].append(projecao_carga["Jan-dez"][row])
                dt = pd.DataFrame(Data_dict)#.astype({"Data_Estudo": "datetime64[ns]", "Data_referencia": "datetime64[ns]"})

                data.append(dt)

        return pd.concat(data).astype(cls.DATA_TYPES_DICT)

    
class PojecaoCargaRaw(AbstractData):
    DATA_DICT = {  # https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/revisoes-quadrimestrais-da-carga
        "Data_Estudo_EPE": "data em que o estudo foi feito",  # usar dia 01/01 do ano
        "Submercado": "nome do subsistema",
        "Data_referencia": "Primeiro dia da carga prevista",
        "Projecao_carga": "Carga média em MW"
    }
    DATA_TYPES_DICT = {
        "Data_Estudo_EPE" : "object", 
        "Submercado" : "int64", 
        "Ano_referencia" : "object", 
        "Jan": "int64",
        "Fev": "int64",
        "Mar": "int64",
        "Abr": "int64",
        "Mai": "int64",
        "Jun": "int64",
        "Jul": "int64",
        "Ago": "int64",
        "Set": "int64",
        "Out": "int64",
        "Nov": "int64",
        "Dez": "int64",
        "Jan-dez": "int64"
    }
    DEFAULT_PATH = "Dados/raw/projecao_carga"
    name = "projecao_carga_raw"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                projecao_carga = pd.read_csv(os.path.join(path, filename), sep=",")
                projecao_carga["Submercado"] = projecao_carga["Submercado"].replace({
                    "Norte": 1, "Nordeste": 2,
                    "Sudeste": 3, "Sul": 4, "SIN": 5})
                data.append(projecao_carga)
        all_data = pd.concat(data).sort_values(["Data_Estudo_EPE", "Submercado", "Ano_referencia"])
        return all_data.astype(cls.DATA_TYPES_DICT)
