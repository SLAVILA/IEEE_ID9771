import pandas as pd
import os

from ..ampere import AmpereDB

from .abstract_data import AbstractData
''' Previsões diárias de Energia natural afluente '''

class AmpereEnaPrevREE(AbstractData):
    DATA_DICT = {
        "data": "O histórico de ENA diária foi calculado com base nos dados de \
        vazão natural para cada posto hidráulico de interesse do SIN e na \
        produtibilidade histórica associada a cada um deles, considerando o \
        horizonte de 01/01/2000 a 30/04/2023.\
        Com adição de dados até 31/05/2024",

    }
    DATA_TYPES_DICT = {f"d+{i}": "float64" for i in range(1, 311)}
    DATA_TYPES_DICT.update({
        "data": "datetime64[ns]",
        "cod_rees": "int",
        "modelo": "object",
        "nome_rees": "object",      
    })
    
    DEFAULT_PATH = "Dados/Ampere/item7-resultados"
    name = "ampere_ena_prevista_rees_raw"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            if modelo.startswith("."):
                continue
            if os.path.isdir(os.path.join(path, modelo)):
                path_temp = os.path.join(path, modelo, "rees")
                print(path_temp)
                raw = cls._load_data_common(path_temp)
                if modelo.endswith("-new"):
                    modelo = modelo[:-4]
                for df, cod, name in raw:
                    df["cod_rees"] = cod[len("ree"):]
                    df["modelo"] = modelo
                    df["nome_rees"] = name
                    data.append(df)
        return pd.concat(data).reset_index(drop=True).astype(cls.DATA_TYPES_DICT)

    @classmethod
    def _load_data_common(cls, path):
        data = []
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                #print(filename)
                n_col = 311
                try:
                    df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None, names=range(n_col))
                except:
                    df = pd.read_csv(os.path.join(path, filename), sep=";", decimal=",", header=None, names=range(n_col), engine='python')
                df = df.rename({i: f"d+{i}" for i in range (1, len(df.columns))}, axis='columns')
                df = df.rename({0: 'data'}, axis='columns') 
                try:
                    df['data'] = pd.to_datetime(df['data'], format="%Y/%m/%d")
                except:
                    try:
                        df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")
                    except:
                        print(path, "skipped")
                        continue

                cod, name, _ = filename.split("_", 2)
                data.append((df, cod, name))
        return data


class UpdatedAmpereEnaPrevREE(AmpereEnaPrevREE):
    name = "ampere_ena_prevista_rees"

    @classmethod
    def _get_data(cls, path):
        updated = AmpereDB().get_table("ampere_ena_prevista_rees")
        old = AmpereEnaPrevREE._get_data(path)
        
        a = old.set_index(["data", "cod_rees", "modelo"])
        b = updated.set_index(["data", "cod_rees", "modelo"])
        unified = pd.concat([a.drop(index=(a.index.intersection(b.index))), b])
        
        columns = updated.columns
        return unified.reset_index()[columns]


class AmpereEnaPrevBacia(AmpereEnaPrevREE):
    DATA_DICT = {
        "data": "",
        
    }
    DATA_TYPES_DICT = {
        "data": "datetime64[ns]",
        "d+1": "float64",
        "d+2": "float64",
        "d+3": "float64",
        "d+4": "float64",
        "d+5": "float64",
        "d+6": "float64",
        "d+7": "float64",
        "d+8": "float64",
        "d+9": "float64",
        "d+10": "float64",
        "d+11": "float64",
        "d+12": "float64",
        "d+13": "float64",
        "d+14": "float64",
        "d+15": "float64",
        "d+16": "float64",
        "d+17": "float64",
        "d+18": "float64",
        "d+19": "float64",
        "d+20": "float64",
        "d+21": "float64",
        "d+22": "float64",
        "d+23": "float64",
        "d+24": "float64",
        "d+25": "float64",
        "d+26": "float64",
        "d+27": "float64",
        "d+28": "float64",
        "d+29": "float64",
        "d+30": "float64",
        "d+31": "float64",
        "d+32": "float64",
        "d+33": "float64",
        "d+34": "float64",
        "d+35": "float64",
        "d+36": "float64",
        "d+37": "float64",
        "d+38": "float64",
        "d+39": "float64",
        "d+40": "float64",
        "d+41": "float64",
        "d+42": "float64",
        "d+43": "float64",
        "d+44": "float64",
        "d+45": "float64",
        "d+46": "float64",
        "d+47": "float64",
        "d+48": "float64",
        "d+49": "float64",
        "d+50": "float64",
        "d+51": "float64",
        "d+52": "float64",
        "d+53": "float64",
        "d+54": "float64",
        "d+55": "float64",
        "d+56": "float64",
        "d+57": "float64",
        "d+58": "float64",
        "d+59": "float64",
        "d+60": "float64",
        "d+61": "float64",
        "d+62": "float64",
        "d+63": "float64",
        "d+64": "float64",
        "d+65": "float64",
        "d+66": "float64",
        "d+67": "float64",
        "d+68": "float64",
        "d+69": "float64",
        "d+70": "float64",
        "d+71": "float64",
        "d+72": "float64",
        "d+73": "float64",
        "d+74": "float64",
        "d+75": "float64",
        "d+76": "float64",
        "d+77": "float64",
        "d+78": "float64",
        "d+79": "float64",
        "d+80": "float64",
        "d+81": "float64",
        "d+82": "float64",
        "d+83": "float64",
        "d+84": "float64",
        "d+85": "float64",
        "d+86": "float64",
        "d+87": "float64",
        "d+88": "float64",
        "d+89": "float64",
        "d+90": "float64",
        "d+91": "float64",
        "d+92": "float64",
        "d+93": "float64",
        "d+94": "float64",
        "d+95": "float64",
        "d+96": "float64",
        "d+97": "float64",
        "d+98": "float64",
        "d+99": "float64",
        "d+100": "float64",
        "d+101": "float64",
        "d+102": "float64",
        "d+103": "float64",
        "d+104": "float64",
        "d+105": "float64",
        "d+106": "float64",
        "d+107": "float64",
        "d+108": "float64",
        "d+109": "float64",
        "d+110": "float64",
        "d+111": "float64",
        "d+112": "float64",
        "d+113": "float64",
        "d+114": "float64",
        "d+115": "float64",
        "d+116": "float64",
        "d+117": "float64",
        "d+118": "float64",
        "d+119": "float64",
        "d+120": "float64",
        "d+121": "float64",
        "d+122": "float64",
        "d+123": "float64",
        "d+124": "float64",
        "d+125": "float64",
        "d+126": "float64",
        "d+127": "float64",
        "d+128": "float64",
        "d+129": "float64",
        "d+130": "float64",
        "d+131": "float64",
        "d+132": "float64",
        "d+133": "float64",
        "d+134": "float64",
        "d+135": "float64",
        "d+136": "float64",
        "d+137": "float64",
        "d+138": "float64",
        "d+139": "float64",
        "d+140": "float64",
        "d+141": "float64",
        "d+142": "float64",
        "d+143": "float64",
        "d+144": "float64",
        "d+145": "float64",
        "d+146": "float64",
        "d+147": "float64",
        "d+148": "float64",
        "d+149": "float64",
        "d+150": "float64",
        "d+151": "float64",
        "d+152": "float64",
        "d+153": "float64",
        "d+154": "float64",
        "d+155": "float64",
        "d+156": "float64",
        "d+157": "float64",
        "d+158": "float64",
        "d+159": "float64",
        "d+160": "float64",
        "d+161": "float64",
        "d+162": "float64",
        "d+163": "float64",
        "d+164": "float64",
        "d+165": "float64",
        "d+166": "float64",
        "d+167": "float64",
        "d+168": "float64",
        "d+169": "float64",
        "d+170": "float64",
        "d+171": "float64",
        "d+172": "float64",
        "d+173": "float64",
        "d+174": "float64",
        "d+175": "float64",
        "d+176": "float64",
        "d+177": "float64",
        "d+178": "float64",
        "d+179": "float64",
        "d+180": "float64",
        "d+181": "float64",
        "d+182": "float64",
        "d+183": "float64",
        "d+184": "float64",
        "d+185": "float64",
        "d+186": "float64",
        "d+187": "float64",
        "d+188": "float64",
        "d+189": "float64",
        "d+190": "float64",
        "d+191": "float64",
        "d+192": "float64",
        "d+193": "float64",
        "d+194": "float64",
        "d+195": "float64",
        "d+196": "float64",
        "d+197": "float64",
        "d+198": "float64",
        "d+199": "float64",
        "d+200": "float64",
        "d+201": "float64",
        "d+202": "float64",
        "d+203": "float64",
        "d+204": "float64",
        "d+205": "float64",
        "d+206": "float64",
        "d+207": "float64",
        "d+208": "float64",
        "d+209": "float64",
        "d+210": "float64",
        "d+211": "float64",
        "d+212": "float64",
        "d+213": "float64",
        "cod_bacia": "object",
        "modelo": "object",
        "nome_bacia": "object",
        
    }
    DEFAULT_PATH = "Dados/Ampere/item7-resultados"
    name = "ampere_ena_prevista_bacias"

    @classmethod
    def _get_data(cls, path):
        data = []
        for modelo in os.listdir(path):
            # print(modelo)
            if modelo.startswith("."):
                continue
            if os.path.isdir(os.path.join(path, modelo)):
                path_temp = os.path.join(path, modelo, "bacias")
                print(path_temp)
                raw = cls._load_data_common(path_temp)
                for df, cod, name in raw:
                    df["cod_bacia"] = cod[len("bacia"):]
                    df["modelo"] = modelo
                    df["nome_bacia"] = name
                    data.append(df)
        return pd.concat(data)#.astype(cls.DATA_TYPES_DICT)