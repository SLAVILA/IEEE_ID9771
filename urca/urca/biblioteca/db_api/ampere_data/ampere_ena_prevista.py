import pandas as pd
import os

''' Previsões diárias de Energia natural afluente '''

from .abstract_data_ampere import AbstractDataAmpere
from .cfsv2 import update_CFSV2_INFORMES
from .utils import *

def get_avaliable_data(path):
    avaliable = {}
    for f in os.listdir(path):
        splited = f.split("_")
        date = dateStr2Timestamp(splited[0])
        acomph = dateStr2Timestamp(splited[1][len("ACOMPH"):])
        date_forecast = splited[2]

        consolidation = -1
        #("ACOMPH"+yesterday_str, date_str)
        if acomph+pd.Timedelta(days=1) == date:
            consolidation = 0
        #("ACOMPH"+date_str,      date_str+"-PSAT"),
        elif date_forecast.endswith("-PSAT"):
            consolidation = 2
        # ("ACOMPH"+date_str,      date_str),
        else:
            consolidation = 1

        if date not in avaliable:
            avaliable[date] = consolidation, f
        else:
            if consolidation > avaliable[date][0]:
                avaliable[date] = consolidation, f
    return avaliable

class EnaPrevCFSV2_REE(AbstractDataAmpere):
    DATA_DICT = {
        "data": "O histórico de ENA diária foi calculado com base nos dados de \
        vazão natural para cada posto hidráulico de interesse do SIN e na \
        produtibilidade histórica associada a cada um deles, considerando o \
        horizonte de 01/01/2000 a 30/04/2023."
    }
    DATA_TYPES_DICT = {f"d+{i}": "float64" for i in range(1, 311)}
    DATA_TYPES_DICT.update({
        "data": "datetime64[ns]",
        "cod_rees": "int",
        "modelo": "object",
        "nome_rees": "object",      
    })
    DEFAULT_PATH = "/Dados/API_Ampere/ENA_PREV_CFSV2-INFORMES"
    name = "ampere_ena_prevista_rees"

    @classmethod
    def _get_data(cls, path):
        avaliable = get_avaliable_data(path)

        formated = []
        for day, (_, file) in avaliable.items():
            raw = pd.read_csv(os.path.join(path, file), sep=';', skipinitialspace=True)
            raw.DATA = pd.to_datetime(raw.DATA, format='%Y%m%d')
            raw = raw[raw.DATA > day]
            formated.append(cls.formatREE(raw))
        updated = pd.concat(formated).reset_index(drop=True).astype(cls.DATA_TYPES_DICT)
        updated = updated.sort_values("data")
        return updated
    
    @classmethod
    def formatREE(cls, df):
        REE = df[df["POSTO/BACIA/REE/SUB"].apply(lambda s: s.startswith("REE_"))].copy()
        REE["cod_rees"] = REE["POSTO/BACIA/REE/SUB"].apply(lambda s: int(s.split("_")[1]))
        # ignorar REE 00
        REE = REE[REE.cod_rees>0]
        REE = REE.groupby("cod_rees").apply(cls.REE2line).reset_index(drop=True)
        REE["modelo"] = "CFSV2-INFORMES"
        REEcode2name = {
            1: 'sudeste', 2: 'sul', 3: 'nordeste',  4: 'norte',
            5: 'itaipu',   6: 'madeira', 7: 'teles-pires', 8: 'belo-monte',
            9: 'manaus-amapa', 10:'parana', 11: 'iguacu', 12: 'paranapanema'
        }
        REE["nome_rees"] = REE["cod_rees"].replace(REEcode2name)
        return REE
    
    @classmethod
    def REE2line(cls, df):
        date_range = pd.date_range(df.DATA.iloc[0], df.DATA.iloc[-1])
        vertical = pd.DataFrame(index=date_range, columns=["ENA"])
        vertical.update(df.set_index("DATA"))

        if vertical.isna().any()["ENA"]:
            raise ValueError(f"found missing data in {df}")
        vertical = vertical.rename(columns={"ENA": df.DATA.iloc[0]-pd.Timedelta(days=1)})
        vertical = vertical.reindex(pd.date_range(df.DATA.iloc[0], df.DATA.iloc[0]+pd.Timedelta(days=309)))
        vertical["d"] = [f"d+{i}" for i in range(1, 311)]
        vertical = vertical.set_index("d")
        vertical.index.name = None

        horizontal = vertical.T
        horizontal["cod_rees"] = df.cod_rees.iloc[0]
        horizontal = horizontal.reset_index(names="data")
        return horizontal
    
    @classmethod
    def _update_data(cls, flux, path):
        update_CFSV2_INFORMES(flux, path)


class EnaHist_REE(AbstractDataAmpere):
    DATA_DICT = {
        "data": "O histórico de ENA diária foi calculado com base nos dados de \
        vazão natural para cada posto hidráulico de interesse do SIN e na \
        produtibilidade histórica associada a cada um deles, considerando o \
        horizonte de 01/01/2000 a 30/04/2023."
    }
    DATA_TYPES_DICT = {
        "01_sudeste": "float64",
        "02_sul": "float64",
        "03_nordeste": "float64",
        "04_norte": "float64",
        "05_itaipu": "float64",
        "06_madeira": "float64",
        "07_teles-pires": "float64",
        "08_belo-monte": "float64",
        "09_manaus-amapa": "float64",
        "10_parana": "float64",
        "11_iguacu": "float64",
        "12_paranapanema": "float64"
    }
    DEFAULT_PATH = "/Dados/API_Ampere/ENA_PREV_CFSV2-INFORMES"
    name = "ampere_ena_hist_rees"

    @classmethod
    def _get_data(cls, path):
        avaliable = get_avaliable_data(path)
        formated = pd.DataFrame(columns=[
            "01_sudeste", "02_sul", "03_nordeste",
            "04_norte", "05_itaipu", "06_madeira",
            "07_teles-pires", "08_belo-monte", "09_manaus-amapa"
            "10_parana", "11_iguacu", "12_paranapanema"
        ], index=pd.Index([], name="data"))

        days = list(avaliable.keys())
        days.sort(reverse=True)
        for day, (_, file) in avaliable.items():
            raw = pd.read_csv(os.path.join(path, file), sep=';', skipinitialspace=True)
            raw.DATA = pd.to_datetime(raw.DATA, format='%Y%m%d')
            raw = raw[raw.DATA <= day]
            temp = cls.formatREE(raw)
            formated = pd.concat([temp.drop(temp.index.intersection(formated.index)), formated])
        return formated.reset_index().sort_values("data").astype(cls.DATA_TYPES_DICT)
    
    @classmethod
    def formatREE(cls, df):
        REE = df[df["POSTO/BACIA/REE/SUB"].apply(lambda s: s.startswith("REE_"))].copy()
        REE["cod_rees"] = REE["POSTO/BACIA/REE/SUB"].apply(lambda s: int(s.split("_")[1]))
        # ignorar REE 00
        REE = REE[REE.cod_rees>0]
        REE["modelo"] = "CFSV2-INFORMES"
        REEcode2name = {
            1: '01_sudeste', 2: '02_sul', 3: '03_nordeste',  4: '04_norte',
            5: '05_itaipu',   6: '06_madeira', 7: '07_teles-pires', 8: '08_belo-monte',
            9: '09_manaus-amapa', 10:'10_parana', 11: '11_iguacu', 12: '12_paranapanema'
        }
        REE = REE.rename(columns={"DATA": "data"})
        data = []
        for rees in REEcode2name:
            d = REE[REE["cod_rees"] == rees]
            data.append(pd.DataFrame({REEcode2name[rees]: d.ENA.values}, index=d.data))
        return pd.concat(data, axis=1)
    
    @classmethod
    def _update_data(cls, flux, path):
        update_CFSV2_INFORMES(flux, path)

    
