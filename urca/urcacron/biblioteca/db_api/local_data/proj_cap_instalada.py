import pandas as pd
import os
from .abstract_data import AbstractData
class ProjCapInstalada(AbstractData):

    DROP_COLUMNS = [
        "Simulada Individual", # igual "Simulada"
        "Filtro Data Ref", # sempre True ou Verdadeiro
        "Ano", # mesma informação de "ano"
        "Data  + (current Year)  Data PMO", #igual "Data  + (current Year) "
        'Data  + (current Year)  Data PMO Ano Mês', #igual 'Data  >=  Data PMO'
        'Data + (current month) Data PMO (cópia)', # igual 'Data + (current month) Data PMO',
        "DataS",         # mesmas datas do "Data", mas com formato diferente (com as horas) 
        "Usina ",        # igual "Usina"
        "Usina (cópia)", # igual "Usina"
        "Usinas",        # igual "Usina"
        "Nome Cadastro Leilão",  # tudo null
        "Nome do Leilão (cópia)", # igual "Nome do Leilão"
        "Referência PMO",   #igual a "Data Estudo PMO"
        "Subsistemas",      # igual "Subsistema "
        "Tipo - Detalhe ",  # igual 'Tipo - Detalhe'
        "Tipo - Detalhe  ", # igual 'Tipo - Detalhe'
        "Tipo Detalhe -EN", # igual "Tipo"
    ]
    
    DATA_DICT = {
        "id_subsistema": "identificador do subsistema",
        "nom_subsistema": "nome do subsistema",
        "nom_bacia": "nome da bacia hidroenergética",
        "nom_rio": "nome do rio",
        "nom_agente": "nome do agente",
        "nom_reservatorio": "nome do reservatorio",
        "cod_usina": "Código da usina nos modelos de otimização",
        "din_instante": "Data",
        "val_geracao": "Valor da Geração, em MWmed",
        "val_disponibilidade": "Valor de disponibilidade, em MWmed ",
        "val_vazaoturbinada": "Vazão turbinada, em m3/s",
        "val_vazaovertida": "Vazão vertida, em m3/s",
        "val_vazaovertidanaoturbinavel": "Vazão vertida não turbinável, em m3/s",
        "val_produtividade": "Valor da produtividade, em MW/(m3/s)",
        "val_folgadegeracao": "Valor da folga de geração, em MWmed",
        "val_energiavertida": "Valor da energia vertida, em MWmed",
        "val_vazaovertidaturbinavel": "Vazão vertida turbinável, em m3/s",
        "val_energiavertidaturbinavel": "Valor da energia vertida turbinável, em MWmed",
    }
    DATA_TYPES_DICT = {
        "Subsistema ": "object",
        "Fonte": "object",
        "Data Estudo PMO": "datetime64[ns]",
        "Simulada": "object",
        "ano": "int64",
        "Ano_referencia": "int64",
        "Cod Aneel": "object",
        "cod_aneel (Consulta SQL personalizada)": "object",
        "cod_anomes (Consulta SQL personalizada)": "float64",
        "cod_dpee": "float64",
        "Data": "object",
        "Data  + (current Year) ": "object",
        "Data  + (current Year)  Data PMO (cópia)": "object",
        "Data  >=  Data PMO": "object",
        "Data + (current month) Data PMO": "object",
        "dsc_qtdug": "object",
        "dsc_quantidadeug": "float64",
        "Filtro Ano Ref": "object",
        "Insumo Parametro Ref": "object",
        "Legenda M&C": "object",
        "nom_leilao": "object",
        "nom_leilao (Consulta SQL personalizada)": "object",
        "Nome do Leilão": "object",
        "Table Name": "object",
        "Tipo": "object",
        "Tipo - Detalhe": "object",
        "Tipo Detalhe - Expansão Não Simualdas": "object",
        "UF": "object",
        "Usina": "object",
        "Count Tipo": "int64",
        "Count Usinas": "int64",
        "Cálculo2": "object",
        "Num Anopmo": "int64",
        "Num Anoreferencia": "int64",
        "Número de registros": "float64",
        "Soma De Potencia": "object",
    }
    DEFAULT_PATH = "Dados/raw/proj_cap_instalada"
    name = "proj_cap_instalada"

    @classmethod
    def _get_data(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(path, filename), decimal='.', sep=",")
                    assert len(df.columns) > 1
                except:
                    df = pd.read_csv(os.path.join(path, filename), decimal='.', sep=";")
                df["Data Estudo PMO"] = df["Data Estudo PMO"].apply(cls.__estudoPMO2date)
                df = df.drop(cls.DROP_COLUMNS, axis=1)
                df["dsc_qtdug"].apply(lambda x: x if pd.isna(x) else int(x.split(" ")[-1].split("-")[-1]))
                df = df.rename(columns={"Number of Records":"Número de registros"})
                data.append(df)
        return pd.concat(data, ignore_index=True).astype(cls.DATA_TYPES_DICT)

    @staticmethod
    def __estudoPMO2date(x):
        return pd.Timestamp(year=x//100, month=x%100, day=1)
