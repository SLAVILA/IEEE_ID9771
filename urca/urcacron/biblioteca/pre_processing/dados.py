#%%
import sys
sys.path.insert(0, '..') # to import modules from parent folder

import datetime
import pandas as pd

from db_api import UrcaDb, Local, BcbDb, AmpereDB
try:
    from .climatologia import Climatologia
except ImportError:
    from climatologia import Climatologia


#%%

class Dados():
    def __init__(self):
        self._dbs = {
            'urca': UrcaDb(),
            'local': Local(),
            'bcb': BcbDb(),
            'self': self
        }
    
    @property
    def data_procesing_dict(self):
        return {
            'PLD_mensal': {
                'orig': 'local',
                'name': 'PLD_mensal',
                'func': 'category_to_column', 
                'args': ('submercado', "data", 'PLD_mensal_medio')},
            'ampere_ENA_atual_REE': { 
                'orig': 'local',
                'name': 'ampere_ENA_hist_REE_new2',
                'func': 'category_to_column', 
                'args': ("REE", "data", "ENA")},
            'ampere_ENA_hist_REE': { 
                'orig': 'local',
                'name': 'ampere_ENA_hist_REE',
                'func': 'category_to_column', 
                'args': ("REE", "data", "ENA")},
            'cmo_semanal': { 
                'orig': 'local',
                'name': 'cmo_semanal',
                'func': 'category_to_column', 
                'args': ("id_subsistema", "din_instante", "val_cmomediasemanal")},
            'cmo_semihorario': { 
                'orig': 'local',
                'name': 'cmo_semihorario',
                'func': 'category_to_column', 
                'args': ("id_subsistema", "din_instante", "val_cmo"),
                'kwargs': {"group_mean": True}},
            'cmo': {
                'orig': ['self', 'self'],
                'name': ['cmo_semanal', 'cmo_semihorario'],
                'func': 'update',
                'kwargs': {"just_update": True}},
            'intercambio_internacional': { 
                'orig': 'local',
                'name': 'intercambio_internacional',
                'func': 'category_to_column', 
                'args': ("nom_paisdestino", "din_instante", "val_intercambiomwmed"),
                'kwargs': {"group_sum": True}},
            'intercambio_nacional': { 
                'orig': 'local',
                'name': 'intercambio_nacional',
                'func': 'category_to_column', 
                'args': ("orig_dest", "din_instante", "val_intercambiomwmed"),
                'kwargs': {"group_sum": True}},
            'ampere_precipitacao_historica_bacias': { 
                'orig': 'local',
                'name': 'ampere_precipitacao_historica_bacias',
                'func': 'category_to_column', 
                'args': ("nome_bacia", "data", "precipitacao")},
            'ampere_precipitacao_historica_rees': { 
                'orig': 'local',
                'name': 'ampere_precipitacao_historica_rees',
                'func': 'category_to_column', 
                'args': ("nome_rees", "data", "precipitacao")},
            'ampere_ASAS': { 
                'orig': 'local',
                'name': 'ampere_ASAS',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_ENA_atual_bacia': { 
                'orig': 'local',
                'name': 'ampere_ENA_atual_bacia',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_ENA_hist_bacia': { 
                'orig': 'local',
                'name': 'ampere_ENA_hist_bacia',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_Frentes_frias': { 
                'orig': 'local',
                'name': 'ampere_Frentes_frias',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_JBN': { 
                'orig': 'local',
                'name': 'ampere_JBN',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_ZCAS': {
                'orig': 'local',
                'name': 'ampere_ZCAS',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_indice_AOO_diario': {
                'orig': 'local',
                'name': 'ampere_indice_AOO_diario',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_indice_IOD': {
                'orig': ['local', 'local'],
                'name': ['ampere_indice_IOD_mensal', 'ampere_indice_IOD_semanal'],
                'func': 'update',
                'args': (['data', 'data'], [{"IODm": "IOD"}, {"IODs": "IOD"}])},
            'ampere_indice_MJO_diario': {
                'orig': 'local',
                'name': 'ampere_indice_MJO_diario',
                'func': 'set_index', 
                'args': ("data",)},
            'ampere_indice_AMO_mensal': {
                'orig': 'local',
                'name': 'ampere_indice_AMO_mensal',
                'func': 'set_index',
                'args': ('data',)},
            'ampere_indice_ONI_mensal': {
                'orig': 'local',
                'name': 'ampere_indice_ONI_mensal',
                'func': 'set_index', 
                'args': ("data",)},
            'piso_teto_pld': {
                'orig': 'local',
                'name': 'piso_teto_pld',
                'func': 'set_index', 
                'args': ("Ano",)},
            'energia_vertida_turbinavel': {
                'orig': 'local',
                'name': 'energia_vertida_turbinavel',
                'func': 'category_to_column', 
                'args': (
                    "id_subsistema",  "din_instante",
                    ['val_geracao',
                     'val_disponibilidade',
                     'val_vazaoturbinada',
                     'val_vazaovertida',
                     'val_vazaovertidanaoturbinavel',
                     'val_produtividade',
                     'val_folgadegeracao',
                     'val_energiavertida',
                     'val_vazaovertidaturbinavel',
                     'val_energiavertidaturbinavel']),
                'kwargs': {"group_sum": True}},
            'ampere_climatologia_ena_bacia': {
                'orig': 'local',
                'name': 'ampere_climatologia_ena_bacia',
                'func': 'climatologia',
                'args': ("bacia\mes",)},
            'ampere_climatologia_ena_ree': {
                'orig': 'local',
                'name': 'ampere_climatologia_ena_ree',
                'func': 'climatologia',
                'args': ("ree\mes",)},
            'ampere_climatologia_precipitacao_bacia': {
                'orig': 'local',
                'name': 'ampere_climatologia_precipitacao_bacia',
                'func': 'climatologia',
                'args': ("bacia\mes",)},
            'ampere_climatologia_precipitacao_ree': {
                'orig': 'local',
                'name': 'ampere_climatologia_precipitacao_ree',
                'func': 'climatologia',
                'args': ("ree\mes",)},
            'ampere_ena_prevista_rees': {
                'orig': 'local',
                'name': 'ampere_ena_prevista_rees',
                'func': 'category_to_column',
                'args': (("cod_rees", "nome_rees", "modelo"), 'data', ['d+'+str(i+1) for i in range(30)])},
            'ampere_ena_prevista_bacias': {
                'orig': 'local',
                'name': 'ampere_ena_prevista_bacias',
                'func': 'category_to_column',
                'args': (("cod_bacia", "nome_bacia", "modelo"), 'data', ['d+'+str(i+1) for i in range(30)])},        
            'ampere_ena_prevista_rees_CFSV2-REFORECAST': {
                'orig': "self",
                'name': "ampere_ena_prevista_rees",
                'func': "select_model",
                'kwargs': {"model": "CFSV2-REFORECAST"}},
            'ampere_ena_prevista_rees_ECMWFENS-REFORECAST': {
                'orig': "self",
                'name': "ampere_ena_prevista_rees",
                "func": "select_model",
                "kwargs": {"model": "ECMWFENS-REFORECAST"}},
            'ampere_ena_prevista_rees_ECMWF-ENSEMBLE-RMV': {
                'orig': "self",
                'name': "ampere_ena_prevista_rees",
                "func": "select_model",
                "kwargs": {"model": "ECMWF-ENSEMBLE-RMV"}},
            'ampere_ena_prevista_rees_ECMWFENS': {
                'orig': ['self', 'self'],
                'name': ['ampere_ena_prevista_rees_ECMWF-ENSEMBLE-RMV', 'ampere_ena_prevista_rees_ECMWFENS-REFORECAST'],
                'func': 'update',
                'kwargs': {"just_update": True}},
            'ampere_ena_prevista_bacias_CFSV2-REFORECAST': {
                'orig': "self",
                'name': "ampere_ena_prevista_bacias",
                "func": "select_model",
                "kwargs": {"model": "CFSV2-REFORECAST"}},
            'ampere_ena_prevista_bacias_ECMWFENS-REFORECAST': {
                'orig': "self",
                'name': "ampere_ena_prevista_bacias",
                "func": "select_model",
                "kwargs": {"model": "ECMWFENS-REFORECAST"}},
            'ampere_ena_prevista_bacias_ECMWF-ENSEMBLE-RMV': {
                'orig': "self",
                'name': "ampere_ena_prevista_bacias",
                "func": "select_model",
                "kwargs": {"model": "ECMWF-ENSEMBLE-RMV"}},
            'ampere_ena_prevista_bacias_ECMWFENS': {
                'orig': ['self', 'self'],
                'name': ['ampere_ena_prevista_bacias_ECMWF-ENSEMBLE-RMV', 'ampere_ena_prevista_bacias_ECMWFENS-REFORECAST'],
                'func': 'update',
                'kwargs': {"just_update": True}},
            'previsao_historica_bacias': {
                'orig': 'local',
                'name': 'previsao_historica_bacias',
                'func': 'category_to_column',
                'args': (("cod_bacia", "nome_bacia", "modelo"), 'data', ['D+'+str(i+1) for i in range(30)])},
            'previsao_historica_bacias_cfsv2': {
                'orig': "self",
                'name': "previsao_historica_bacias",
                "func": "select_model",
                "kwargs": {"model": "cfsv2"}},
            'previsao_historica_bacias_ecmwfens': {
                'orig': "self",
                'name': "previsao_historica_bacias",
                "func": "select_model",
                "kwargs": {"model": "ecmwfens"}},
            'previsao_historica_rees': { 
                'orig': 'local',
                'name': 'previsao_historica_rees',
                'func': 'category_to_column', 
                'args': (('cod_rees', 'nome_rees', 'modelo'), "data", ['D+'+str(i+1) for i in range(30)])},
            'previsao_historica_rees_cfsv2': {
                'orig': "self",
                'name': "previsao_historica_rees",
                "func": "select_model",
                "kwargs": {"model": "cfsv2"}},
            'previsao_historica_rees_ecmwfens': {
                'orig': "self",
                'name': "previsao_historica_rees",
                "func": "select_model",
                "kwargs": {"model": "ecmwfens"}},
            'demanda_maxima': {
                'orig': 'local',
                'name': 'demanda_maxima',
                'func': 'category_to_column',
                'args': ('Subsistema', 'Din Instante', 'Demanda Maxima')},
            'geracao_usinas_despachadas': {
                'orig': 'local',
                'name': 'geracao_usinas_despachadas',
                'func': 'category_to_column',
                'args': (('Subsistema', 'Tipo'), 'Data', 'Geração')},
        } 
    
    def process_data(self, orig, name, func, args=(), kwargs={}):
        if isinstance(name, tuple) or isinstance(name, list):
            raw_data = [self._dbs[orig[i]].get_table(name[i]) for i in range(len(name))]
        else:
            raw_data = self._dbs[orig].get_table(name)
        return self.functions[func](raw_data, *args, **kwargs)
    
    def get_table(self, *args, **kwargs):
        table = self.get_data(*args, **kwargs)
        table.columns = table.columns.get_level_values(1)
        return table
    
    def get_data(self, table_name, data_inic=datetime.date(2014, 1, 1), data_fim=datetime.date(2023, 12, 31)):
        process_dict = self.data_procesing_dict[table_name]
        data_processed = self.process_data(**process_dict)
        try:
            data = data_processed[
                (data_processed.index >= data_inic.strftime("%Y/%m/%d")) & 
                (data_processed.index <= data_fim.strftime("%Y/%m/%d"))
            ]
        except:
            data = data_processed[data_inic: data_fim]
        data.columns = pd.MultiIndex.from_tuples([(table_name, c) for c in data.columns])
        data = data.reindex(pd.date_range(data_inic, data_fim))
        data.index.names = ['data']
        data.index = data.index.astype('datetime64[ns]')
        return data
    
    @property
    def functions(self):
        return {
            'category_to_column': self.category_to_column,
            'set_index': lambda t, *args: t.set_index(*args),
            'climatologia': lambda x, idx: Climatologia(x.set_index(idx)), 
            'raw': lambda x: x,
            'update': self.update_data,
            'select_model': self.select_model,
        }
    
    @property
    def avalible(self):
        return list(self.data_procesing_dict.keys())
        
    @staticmethod
    def select_model(df, model):
        columns = []
        for c in list(df.columns):
            #print(c)
            s = c.split()
            m = s[2]
            if m == model:
                columns.append((" ".join(s[:2]+s[3:])))
                df = df.rename(columns={c: columns[-1]})
        return df[columns]
    
    @staticmethod
    def update_data(dfs, index_col=None, rename_col={}, just_update=False):
        if rename_col:
            dfs = [dfs[i].rename(columns=rename_col[i])
                   for i in range(len(dfs))]
        if index_col:
            dfs = [dfs[i].set_index(index_col[i]) 
                   for i in range(len(dfs))]
                
        if just_update:
            results = dfs[0]
            for i in range(1, len(dfs)): 
                results.update(dfs[i], overwrite=False)
        else:
            results = pd.concat(dfs, join='outer')
            results = results[~results.index.duplicated(keep='last')]
        return results
    
    @staticmethod
    def category_to_column(df, category_col, index_col, value_col, group_sum=False, group_mean=False):
        if isinstance(category_col, tuple) or isinstance(category_col, list):
            grouped = df.groupby(list(category_col))
        else:
            grouped = df.groupby(category_col)
            
        if isinstance(value_col, tuple):
            value_col = list(value_col)

        if isinstance(value_col, list):
            dfs = []
            for key, val in grouped:
                if group_sum or group_mean:
                    val = val[value_col+[category_col, index_col]].copy()
                    val[index_col] = val[index_col].dt.date
                    if group_sum:
                        val = val.groupby([category_col, index_col]).sum().reset_index()
                    if group_mean:
                        val = val.groupby([category_col, index_col]).mean().reset_index()

                reindexed = val.set_index(index_col)[value_col]
                if isinstance(category_col, tuple) or isinstance(category_col, list):
                    renamed = reindexed.rename(columns={v: " ".join([str(k) for k in key])+" "+v for v in value_col})
                else:
                    renamed = reindexed.rename(columns={v: key+" "+v for v in value_col})
                #if not renamed.index.is_unique:
                    #print(renamed[renamed.index.duplicated(keep=False)].iloc[0][[list(renamed.columns)[0]]])
                renamed = renamed[~renamed.index.duplicated(keep='first')]
                dfs.append(renamed.drop_duplicates())
            data = pd.concat(dfs, axis=1)
        elif isinstance(value_col, str):
            dfs = [v.set_index(index_col)[value_col].rename(k) for k, v in grouped]
            data = pd.concat(dfs, axis=1)
            if group_sum:
                data = data.reset_index()
                data[index_col] = data[index_col].dt.date
                data = data.groupby(index_col).sum()
        else:
            raise ValueError(f"value_col must be a string or list of string. value_col is {type(value_col)} ({value_col})")
        return data


class DadosInterpolados():
    def __init__(self):
        self._dados = Dados()
        
    @property
    def data_procesing_dict(self):
        return {
            'PLD_mensal': {
                "func": "ffill"},
            'ampere_ENA_atual_REE': {
                "func": "ok"},
            'ampere_ENA_hist_REE': {
                "func": "ok"},
            'cmo': {
                "func": "max_range",
                "kwargs": {"min_val": 0, "max_val": 3000}},
            'intercambio_internacional': {
                "func": "interpolate"},
            'intercambio_nacional': {
                "func": "interpolate"},
            'ampere_precipitacao_historica_bacias': {
                "func": "ok"},
            'ampere_precipitacao_historica_rees': {
                "func": "ok"},
            'ampere_ASAS': {
                "func": "ok"},
            'ampere_ENA_atual_bacia': {
                "func": "ok"},
            'ampere_ENA_hist_bacia': {
                "func": "ok"},
            'ampere_Frentes_frias': {
                "func": "ok"},
            'ampere_JBN': {
                "func": "ok"},
            'ampere_ZCAS': {
                "func": "drop",
                "kwargs": {"t": ('ampere_ZCAS', "Posicao_climatologica")}},
            'ampere_indice_AOO_diario': {
                "func": "ok"},
            'ampere_indice_IOD': {
                "func": "interpolate"},
            'ampere_indice_MJO_diario': {
                "func": "ok"},
            'ampere_indice_AMO_mensal': {
                "func": "interpolate"},
            'ampere_indice_ONI_mensal': {
                "func": "interpolate"},
            'piso_teto_pld': {
                "func": "ffill"},
            'energia_vertida_turbinavel': {
                "func": "interpolate"},
            'ampere_climatologia_ena_bacia': {
                "func": "ok"},
            'ampere_climatologia_ena_ree': {
                "func": "ok"},
            'ampere_climatologia_precipitacao_bacia': {
                "func": "ok"},
            'ampere_climatologia_precipitacao_ree': {
                "func": "ok"},
            'ampere_ena_prevista_rees_CFSV2-REFORECAST': {
                "func": "interpolate"},
            'ampere_ena_prevista_rees_ECMWFENS': {
                "func": "interpolate"},
            'ampere_ena_prevista_bacias_CFSV2-REFORECAST': {
                "func": "interpolate"},
            'ampere_ena_prevista_bacias_ECMWFENS-REFORECAST': {
                "func": "interpolate"},
            'previsao_historica_bacias_ecmwfens': {
                "func": "interpolate"},
            'previsao_historica_bacias_cfsv2': {
                "func": "interpolate"},
            'previsao_historica_rees_ecmwfens': {
                "func": "interpolate"},
            'previsao_historica_rees_cfsv2': {
                "func": "interpolate"},
            'demanda_maxima': {
                "func": "interpolate"},
            'geracao_usinas_despachadas': {
                "func": "filter_geracao_usinas_despachadas"},
        }
    
    @property
    def avalible(self):
        return list(self.data_procesing_dict.keys())

    @property
    def functions(self):
        return {
            'ffill': lambda d: d.ffill(),
            'ok': lambda d: d,
            'interpolate': DadosInterpolados.interpolate,
            'max_range': DadosInterpolados.max_range,
            'filter_geracao_usinas_despachadas': DadosInterpolados.filter_geracao_usinas_despachadas,
            'drop': lambda d, t: d.drop(t, axis=1)
        }
    
    @staticmethod
    def interpolate(d):
        return d.interpolate().dropna(axis=1, how="all")
    
    @staticmethod
    def max_range(d, min_val, max_val):
        d[d<min_val] = None
        d[d>max_val] = None
        return DadosInterpolados.interpolate(d)
    
    @staticmethod
    def filter_geracao_usinas_despachadas(d):
        d = d[d.columns[d.max()>1000]]
        return DadosInterpolados.interpolate(d).fillna(0)
        
    
    def get_data(self, table_name, data_inic=datetime.date(2014, 1, 1), data_fim=datetime.date(2023, 12, 31)):
        process_dict = self.data_procesing_dict[table_name]
        dados = self._dados.get_data(table_name, data_inic=data_inic, data_fim=data_fim)
        #print(dados.dropna(axis=1, how="all").isna().any(axis=1).sum())
        #print(dados[dados.isna().any(axis=1)])
        if "kwargs" in process_dict:
            return self.functions[process_dict["func"]](dados, **process_dict["kwargs"])
        return self.functions[process_dict["func"]](dados)

if __name__ == "__main__":
    from termcolor import colored

    interpol = DadosInterpolados()
    data = []
    var = [
        #"PLD_mensal",
        "ampere_ENA_atual_REE",
        #"ampere_ENA_hist_REE"
        #"ampere_ENA_atual_bacia",
        #"ampere_ENA_hist_bacia
        "cmo",
        #"intercambio_internacional",
        #"intercambio_nacional",
        #"ampere_precipitacao_historica_bacias
        #"ampere_precipitacao_historica_rees",
        #"ampere_ASAS",
        #"ampere_Frentes_frias",
        #"ampere_JBN",
        #"ampere_ZCAS",
        #"ampere_indice_AOO_diario",
        #"ampere_indice_IOD",
        #"ampere_indice_MJO_diario",
        #"ampere_indice_AMO_mensal",
        "ampere_indice_ONI_mensal",
        #"piso_teto_pld",
        #"energia_vertida_turbinavel",
        #ampere_climatologia_ena_bacia
        #"ampere_climatologia_ena_ree",
        #ampere_climatologia_precipitacao_bacia
        #"ampere_climatologia_precipitacao_ree",
        "ampere_ena_prevista_rees_CFSV2-REFORECAST",
        #ampere_ena_prevista_rees_ECMWFENS
        #ampere_ena_prevista_bacias_CFSV2-REFORECAST
        #ampere_ena_prevista_bacias_ECMWFENS-REFORECAST
        #previsao_historica_bacias_ecmwfens
        #previsao_historica_bacias_cfsv2
        #previsao_historica_rees_ecmwfens
        #"previsao_historica_rees_cfsv2",
        #"demanda_maxima",
        #"geracao_usinas_despachadas"
    ]
    for t in var:
        print(t)
        data.append(
            interpol.get_data(
            t,
            data_inic=datetime.date(2014, 1, 1), 
            data_fim=datetime.datetime.today()
        ).loc["2015/01/01":].astype("float"))
    alld = pd.concat(data, axis=1)
    alld.to_csv("filtered_ree_data.csv")
    #display(alld)

if False:
    if __name__ == "__main__":
        funcs = {}
        d = DadosInterpolados().data_procesing_dict
        for t in d:
            if d[t]["func"] not in funcs:
                funcs[d[t]["func"]] = []
            funcs[d[t]["func"]].append(t)

        data = {}
        for f in funcs:
            for c in funcs[f]:
                data[c] = Dados().get_data(c)
                #print(data[c].shape[1])#, "\t", c)

    if __name__ == "__main__":
        from termcolor import colored

        interpol = DadosInterpolados()
        data = []
        var = [
            "PLD_mensal",
            "ampere_ENA_atual_REE",
            "ampere_ENA_hist_REE",
            "ampere_ENA_atual_bacia",
            "ampere_ENA_hist_bacia",
            "cmo",
            "intercambio_internacional",
            "intercambio_nacional",
            "ampere_precipitacao_historica_bacias",
            "ampere_precipitacao_historica_rees",
            "ampere_ASAS",
            "ampere_Frentes_frias",
            "ampere_JBN",
            "ampere_ZCAS",
            "ampere_indice_AOO_diario",
            "ampere_indice_IOD",
            "ampere_indice_MJO_diario",
            "ampere_indice_AMO_mensal",
            "ampere_indice_ONI_mensal",
            "piso_teto_pld",
            "energia_vertida_turbinavel",
            "ampere_climatologia_ena_bacia",
            "ampere_climatologia_ena_ree",
            "ampere_climatologia_precipitacao_bacia",
            "ampere_climatologia_precipitacao_ree",
            "ampere_ena_prevista_rees_CFSV2-REFORECAST",
            "ampere_ena_prevista_rees_ECMWFENS",
            "ampere_ena_prevista_bacias_CFSV2-REFORECAST",
            "ampere_ena_prevista_bacias_ECMWFENS-REFORECAST",
            "previsao_historica_bacias_ecmwfens",
            "previsao_historica_bacias_cfsv2",
            "previsao_historica_rees_ecmwfens",
            "previsao_historica_rees_cfsv2",
            "demanda_maxima",
            "geracao_usinas_despachadas"
        ]
        for t in var:
            data.append(DadosInterpolados().get_data(t, data_inic=datetime.date(2015, 1, 1), data_fim=datetime.date(2023, 1, 1)))


    if __name__ == "__main__":
        data[0].size

    if __name__ == "__main__":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_rescaled = pd.DataFrame(scaler.fit_transform(alld), index=alld.index)
        display(data_rescaled)
        
        #data_rescaled.to_csv("filtered_ree_data_pca.csv")
        
        pca = PCA().fit(data_rescaled)
        
        import pickle as pk
        import numpy as np
        import json
        
        y = np.cumsum(pca.explained_variance_ratio_)
        idx = {i: int(np.searchsorted(y, i, side="left")) for i in [0.8, 0.90, 0.95, 0.99]}
        
        #with open("filtered_ree_pca_threshold.json", "w") as f:
        #    json.dump(idx, f)

        #with open("filtered_ree_pca.pkl","wb") as f:
        #    pk.dump(pca, f)


    idx

    if __name__ == "__main__":
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()

        plt.ylim(0.0, 105)
        plt.xlim(0  , y.size)

        plt.plot(y*100)#, marker='o', linestyle='--', color='b')

        plt.xlabel('Número de componentes')
        plt.xticks(list(idx.values())+[y.size]) #change from 0-based array index to 1-based human-readable label
        plt.xticks(rotation=90)
        plt.ylabel('Variância acumulada (%)')
        plt.title('Número de componentes necessários para variância')

        #labels = [item.get_text() for item in ax.get_xticklabels()]

        for i in idx:
            plt.axvline(x=idx[i], color='r', linestyle='--')
            plt.text(idx[i], 5, f"{i*100}% variância", rotation=90)

        plt.show()

    if __name__ == "__main__":
        from termcolor import colored

        d = Dados()
        interpol = DadosInterpolados()
        tables = []
        aaaaa = list(interpol.data_procesing_dict.keys())
        aaaaa.sort()
        for i in aaaaa:
            if len(interpol.data_procesing_dict[i]) == 0:
                print(colored(f"VER\t - {i}", "red"))
            else:
                print(colored(f"OK\t - {i}", "green"))
            #print(colored(f"Não usar - {t}", "light_grey"))
        for i in d.data_procesing_dict:
            if i not in interpol.data_procesing_dict:
                print(colored(f"VER\t - {i}", "red"))

    if __name__ == "__main__":
        from IPython.display import display

        d = DadosInterpolados()
        #d = Dados()
        a = d.get_data(
            "ampere_ZCAS", 
            data_inic=datetime.date(2014, 1, 1), 
            data_fim=datetime.date(2023, 1, 1)
        )
        with pd.option_context('display.min_rows', 10):
            #a = a[a.columns[a.max()>1000]]
            display(a)#.plot(legend=False))
            b = a.loc["2015/01/01":].dropna(axis=1, how="all")
            display(b)
            missing = b[b.isna().any(axis=1)]
            
            display(missing)
            display(b.columns[b.count()!=len(b)].get_level_values(1))
            display(b[b.columns[b.count()!=len(b)]].count())
            
            """
            AAAAAAA = {}
            for t, c in missing.columns[missing.count()!=len(missing)]:
                d = c.split()[-1]
                if d not in AAAAAAA:
                    AAAAAAA[d] = 0
                AAAAAAA[d] += 1
            display(AAAAAAA)
            """

    if __name__ == "__main__":
        from IPython.display import display
        with pd.option_context('display.max_rows', None):
            not_complete = all_data.count() != len(all_data)
            display(pd.DataFrame({
                "tem": all_data[not_complete.index[not_complete]].count(),
                "falta": all_data[not_complete.index[not_complete]].count()*-1+len(all_data)}))

    if __name__ == "__main__":
        display(not_complete[~not_complete].index.get_level_values(0).unique())

    if __name__ == "__main__":
        d = Dados()
        tables = []
        avalible = list(d.avalible)
        avalible.sort()
        for t in avalible:
            #try:
                print(t)
                a = d.get_data(
                    t, 
                    data_inic=datetime.date(2014, 1, 1), 
                    data_fim=datetime.date(2023, 1, 1)
                )
                
                #'''
                if len(a.columns)<100:
                    cleaned = a.dropna()
                    print(t)
                    print('\t', cleaned.index.min(), cleaned.index.max())
                    tables.append(a)
                #'''
                """
                duplicated = a.index.duplicated()
                if sum(duplicated):
                    print(t, "has duplicated")
                    print(a.index[duplicated])
                    print(a)
                else:
                    tables.append(a)
                """
            #except Exception as e:
            #    print(t)
            #    print("\t", e)
        #print(tables)
        all_data = pd.concat(tables, axis=1).sort_index()
        print()
        #print(all_data)
        print(all_data.memory_usage(deep=True))
        #a = d.get_data("ampere_indice_ONI_mensal")#.reset_index()
        #a["data"] = a["din_instante"].dt.date
        #a = a.drop(columns="din_instante", level=0)

    if __name__ == "__main__":
        print(Dados().get_data("ampere_indice_IOD").plot())

    if __name__ == "__main__":
        print(Dados().get_data("ampere_ASAS"))

    if __name__ == "__main__":
        a = [Dados().get_data("ampere_indice_IOD_semanal"),
            Dados().get_data("ampere_indice_MJO_diario")]
        print(pd.concat(a, axis=1))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_ENA_hist_REE"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_ENA_hist_bacia"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_precipitacao_historica_bacias"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_precipitacao_historica_rees"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_indice_IOD_semanal"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_indice_ONI_trimestral"))

    if __name__ == "__main__":
        print(Dados().get_data("energia_vertida_turbinavel"))

    if __name__ == "__main__":
        AAA = ((Dados().get_data("ampere_ena_prevista_rees")).columns.get_level_values(1))
        BBB = str(AAA)
        print(BBB)

    if __name__ == "__main__":
        a = Dados().get_data("ampere_previsao_climatica_rees")
        modelos = {} 
        for t, c in list(a.columns):
            #print(c)
            modelo = c.split()[2]
            if modelo not in modelos:
                modelos[modelo] = []
            modelos[modelo].append((t, c))
        print(a[modelos["ECMWF"]])
        print(a[modelos["NCEPv2"]])

    if __name__ == "__main__":
        a = Dados().get_data("previsao_historica_bacias")
        modelos = {} 
        for t, c in list(a.columns):
            #print(c)
            modelo = c.split()[2]
            if modelo not in modelos:
                modelos[modelo] = []
            modelos[modelo].append((t, c))
        print(modelos.keys())

    if __name__ == "__main__":
        a = Dados().get_data("ampere_ena_prevista_bacias")
        modelos = {} 
        for t, c in list(a.columns):
            #print(c)
            modelo = c.split()[2]
            if modelo not in modelos:
                modelos[modelo] = []
            modelos[modelo].append((t, c))
        print(a[modelos["CFSV2-REFORECAST"]])
        print(a[modelos["ECMWFENS-REFORECAST"]])



    if __name__ == "__main__":
        print(Dados().get_data("previsao_historica_rees"))

    if __name__ == "__main__":
        print(Dados().get_data("previsao_historica_bacias"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_previsao_climatica_bacias"))

    if __name__ == "__main__":
        print(Dados().get_data("ampere_previsao_climatica_rees"))

    if __name__ == "__main__":
        a = d.avalible
        a.sort()
        a

    if __name__ == "__main__":
        for col in all_data["geracao_usinas_despachadas"]:
            data = all_data["geracao_usinas_despachadas"][col]
            print(col)
            print("\t", data.first_valid_index(), data.last_valid_index())

    if __name__ == "__main__":
        from IPython.display import display
        with pd.option_context('display.max_rows', 10):
            display(all_data["PLD_mensal"][all_data["PLD_mensal"].isna().any(axis=1)])

    if __name__ == "__main__":

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 10))
        all_data["geracao_usinas_despachadas"].interpolate().plot(ax=ax)
        all_data.reset_index().plot(x='data', y='geracao_usinas_despachadas', ax=ax, style="x")

    if __name__ == "__main__":
        all_data.infer_objects(copy=False).interpolate()

    if __name__ == "__main__":
        a = d.get_data(
            "previsao_historica_rees", 
            data_inic=datetime.date(2014, 1, 1), 
            data_fim=datetime.date(2023, 12, 31)
        )
        a

    if __name__ == "__main__":
        pd.set_option('display.max_columns', 100)
        all_data

    if __name__ == "__main__":
        print(all_data.bfill().ffill().fillna(0).memory_usage(deep=True).sum())
        print(all_data.memory_usage(deep=True).sum())

    if __name__ == "__main__":
        all_data.dtypes.groupby(all_data.dtypes).count()

    if __name__ == "__main__":
        all_data[all_data.columns[all_data.dtypes=="object"]].dropna().drop_duplicates()

    if __name__ == "__main__":
        all_data[("ampere_indice_IOD_mensal", "IODm")].str.replace(',', '.').replace('-', None).astype("float")

    if __name__ == "__main__":
        t = d.get_data(d.avalible[2])
        t.columns = pd.MultiIndex.from_tuples([("a", g) for g in t.columns])
        t

    if __name__ == "__main__":
        import datetime
        d = Dados()
        #fazer retornar climatologia?? 
        tables = [
            'ampere_climatologia_ena_bacia',
            'ampere_climatologia_ena_ree',
            'ampere_climatologia_precipitacao_bacia',
            'ampere_climatologia_precipitacao_ree'
        ]
        a = []
        for table in tables:
            print(table)
            b = d.get_data(table)
            print(b)
            #b(estou terminando de ajustar os códigos das redes neurais para realizar os testes multivariados).plot(figsize=(20, 10))
