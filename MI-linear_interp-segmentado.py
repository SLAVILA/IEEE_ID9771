# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import os

from Documents.URCA.db_api import BcbDb
from Documents.URCA.db_api import UrcaDb
from Documents.URCA.db_api import Local
BCB = BcbDb()
local = Local()
urca = UrcaDb()

# +
min = datetime.datetime(2014, 12, 16)
max = datetime.datetime(2023, 4, 28)

plt.style.use("seaborn-whitegrid")
drop_columns = ["Unnamed: 0", "submercado", "M", "H", "h", "h_cresc", "volume", "expiracao", "produto"]


def get_df(path, nome):
    dff = pd.read_csv(path)#.drop(drop_columns, axis=1)
    if "result" in dff.columns:
        dff = dff.rename(columns={"result" : str(nome)}).astype({'data' : 'datetime64[ns]'})
    else:
        dff = dff.rename(columns={"VWAP" : str(nome)}).astype({'data' : 'datetime64[ns]'})
    dff = dff[["data", str(nome)]]
    mask = (dff['data'] >= min) & (dff['data'] <= max)
    df_temp = dff.loc[mask]
    df_temp = df_temp.set_index('data')
    return df_temp


def data_path(path):
    paths = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            path_temp = os.path.join(path, filename)
            paths.append((path_temp))
    return paths


def get_base(paths):
    dfs = []
    for i in range(len(paths)):
        df = get_df(paths[i], paths[i][:-4])
        dfs.append(df)
    return dfs


def get_var(VAR, variavel=None):
    full_idx = pd.date_range('2014-12-01', max, name='data')
    VAR = VAR.set_index('data')
    VAR = VAR.reindex(full_idx, method='ffill')
    VAR = VAR.reset_index()

    mask = (VAR['data'] >= min) & (VAR['data'] <= max)
    df_temp = VAR.loc[mask]
    df_temp = df_temp.set_index('data')
    if "valor" in df_temp.columns:
        df_temp = df_temp.rename(columns={'valor': variavel})
    return df_temp


def get_GPR(path):
    df_temp = pd.read_csv(path, parse_dates=['date'], dayfirst=True, decimal=',').rename(columns={"date" : "data"})
    df_temp = df_temp.set_index('data')
    return df_temp


def get_evt(evt):
    
    return evt


def get_proj_carga(suf):
    '''
    Projeção de carga considera o ano de estudo e 4 anos a frente,
    para o feature engineering a projeção de carga foi separada em M0 até M4
    para o ano de 2011 por exemplo, M0 é a projeção feita em 2011 para 2011
    M1 é a projeção para 2012 feita em 2011 e assim por diante
    '''
    proj_carga = local.get_table("projecao_carga")
    proj_c = proj_carga[proj_carga['Submercado'] == suf][['Data_Estudo', 'Data_referencia', 'Projecao_carga']]
    proj_c['Data_referencia'] = pd.to_datetime(proj_c['Data_referencia'])
    proj_c = proj_c[proj_c['Data_Estudo'].dt.year >= 2011]
    proj_c = proj_c.sort_values(by='Data_Estudo')
    proj_c = proj_c.rename(columns={'Data_referencia' : 'data'})

    proj_carga_M0 = proj_c[proj_c["Data_Estudo"].dt.year == proj_c['data'].dt.year].rename(columns= {'Projecao_carga': 'Proj_c_M0'}).reset_index(drop=True).sort_values(by='data').astype({'data': 'datetime64[ns]'})
    proj_carga_M1 = proj_c[proj_c["Data_Estudo"].dt.year == proj_c['data'].dt.year - 1].rename(columns= {'Projecao_carga': 'Proj_c_M1'}).reset_index(drop=True).sort_values(by='data').astype({'data': 'datetime64[ns]'})
    proj_carga_M2 = proj_c[proj_c["Data_Estudo"].dt.year == proj_c['data'].dt.year - 2].rename(columns= {'Projecao_carga': 'Proj_c_M2'}).reset_index(drop=True).sort_values(by='data').astype({'data': 'datetime64[ns]'})
    proj_carga_M3 = proj_c[proj_c["Data_Estudo"].dt.year == proj_c['data'].dt.year - 3].rename(columns= {'Projecao_carga': 'Proj_c_M3'}).reset_index(drop=True).sort_values(by='data').astype({'data': 'datetime64[ns]'})
    proj_carga_M4 = proj_c[proj_c["Data_Estudo"].dt.year == proj_c['data'].dt.year - 4].rename(columns= {'Projecao_carga': 'Proj_c_M4'}).reset_index(drop=True).sort_values(by='data').astype({'data': 'datetime64[ns]'})
    
    proj_carga_M0 = proj_carga_M0.drop('Data_Estudo', axis=1)
    proj_carga_M1 = proj_carga_M1.drop('Data_Estudo', axis=1)
    proj_carga_M2 = proj_carga_M2.drop('Data_Estudo', axis=1)
    proj_carga_M3 = proj_carga_M3.drop('Data_Estudo', axis=1)
    proj_carga_M4 = proj_carga_M4.drop('Data_Estudo', axis=1)
    #df = pd.concat([proj_carga_M0, proj_carga_M1, proj_carga_M2, proj_carga_M3, proj_carga_M4], axis=1)
    proj_c_M0 = get_var(proj_carga_M0, "Proj_c_M0")#.set_index('Data_Estudo', drop=True)
    proj_c_M1 = get_var(proj_carga_M1, "Proj_c_M1")
    proj_c_M2 = get_var(proj_carga_M2, "Proj_c_M2")
    proj_c_M3 = get_var(proj_carga_M3, "Proj_c_M3")
    proj_c_M4 = get_var(proj_carga_M4, "Proj_c_M4")
    return (proj_c_M0, proj_c_M1, proj_c_M2, proj_c_M3, proj_c_M4)


def get_earm():
    '''
    Separa o earm em Dataframes por subsistema
    '''
    earm = pd.merge(urca.get_table("subsistema").rename(columns={"nome": "nome_regiao","id_subsistema":"regiao"}), urca.get_table("historico_hidrologia"), on="regiao")
    earm = earm[['nome_regiao', 'data','earm']].astype({'data' : 'datetime64[ns]'})
    earm = earm[earm['data'].dt.year >= 2014]

    earm_SIN = earm[earm["nome_regiao"] == 'SIN'].rename({'earm' : 'earm_SIN'}, axis=1).drop("nome_regiao", axis=1).set_index('data', drop=True)
    earm_N = earm[earm["nome_regiao"] == 'Norte'].rename({'earm' : 'earm_N'}, axis=1).drop("nome_regiao", axis=1).set_index('data', drop=True)
    earm_NE = earm[earm["nome_regiao"] == 'Nordeste'].rename({'earm' : 'earm_NE'}, axis=1).drop("nome_regiao", axis=1).set_index('data', drop=True)
    earm_S = earm[earm["nome_regiao"] == 'Sul'].rename({'earm' : 'earm_S'}, axis=1).drop("nome_regiao", axis=1).set_index('data', drop=True)
    earm_SE = earm[earm["nome_regiao"] == 'Sudeste'].rename({'earm' : 'earm_SE'}, axis=1).drop("nome_regiao", axis=1).set_index('data', drop=True)
    return (earm_SIN, earm_N, earm_NE, earm_S, earm_SE)


def get_cmo(CMO, suf):
    #CMO = local.get_table('cmo_semanal')
    CMO_temp = CMO[CMO['id_subsistema'] == suf][['din_instante', 'val_cmomediasemanal']]
    CMO_temp['din_instante'] = pd.to_datetime(CMO_temp['din_instante'])
    CMO_temp = CMO_temp[CMO_temp['din_instante'].dt.year >= 2019]
    CMO_temp = CMO_temp.sort_values(by='din_instante')
    CMO_temp = CMO_temp.rename(columns={'din_instante' : 'data', 'val_cmomediasemanal': 'valor'})
    cmo_temp = get_var(CMO_temp, "CMO_"+str(suf))
    return cmo_temp


def get_oni(oni):
    ''' 
    estendendo o indice para todos os dias de cada mes,
    para verificar correlacao
    '''
    full_idx = pd.date_range('2014-12-01', max, name='data')
    oni = oni.set_index('data')
    oni = oni.reindex(full_idx, method='ffill')
    oni = oni.reset_index()

    mask = (oni['data'] >= min) & (oni['data'] <= max)
    df_temp = oni.loc[mask]
    df_temp = df_temp.set_index('data')
    return df_temp


def MM_df(df, SMA):
    df_MM5 = df['ENA_hist'].rolling(SMA).mean().rename("ENA_hist_MM5")
    df_MM7 = df['ENA_hist'].rolling(SMA+2).mean().rename("ENA_hist_MM7")
    df_MM15 = df['ENA_hist'].rolling(SMA*3).mean().rename("ENA_hist_MM15")
    df_MM30 = df['ENA_hist'].rolling(SMA*6).mean().rename("ENA_hist_MM30")
    df_MM60 = df['ENA_hist'].rolling(SMA*12).mean().rename("ENA_hist_MM60")
    df_MM90 = df['ENA_hist'].rolling(SMA*18).mean().rename("ENA_hist_MM90")
    df_temp = pd.concat([df, df_MM5, df_MM7, df_MM15, df_MM30, df_MM60, df_MM90], axis=1)
    return df_temp
    
    
def shift_df(df, period, freq):
    df_shifted_s1 = df.shift(periods=period, freq=freq).add_suffix("_s30")
    df_shifted_s2 = df.shift(periods=period*2, freq=freq).add_suffix("_s60")
    df_shifted_s3 = df.shift(periods=period*3, freq=freq).add_suffix("_s90")
    df_shifted_s4 = df.shift(periods=period*4, freq=freq).add_suffix("_s120")
    df_temp = pd.concat([df, df_shifted_s1, df_shifted_s2, df_shifted_s3, df_shifted_s4], axis=1)
    return df_temp


def get_ENA(ena_path):
    df_temp = pd.read_csv(ena_path).rename(columns={"hist" : "ENA_hist", "diff" : "ENA_hist-clim"}).astype({'data' : 'datetime64[ns]'})
    df_temp = df_temp.set_index('data')
    df_temp = MM_df(df_temp, 5)
    df_temp = shift_df(df_temp, 30, "D")
    df_temp = df_temp.dropna()
    return df_temp


def get_reversed_listofnames(df):
    listee = list(df.columns)
    reversed_strings = [x[::-1] for x in listee][::-1]
    reversed_strings.sort()
    reversed_strings
    listee = [x[::-1] for x in reversed_strings][::-1]
    return listee


def get_listofnames(df):
    listee = list(df.columns)
    return listee




# +
paths_linear_interp = data_path("linear_interp")
df_linear_interp = get_base(paths_linear_interp)
#df_pentada
ena = get_ENA("ENA_SIN.csv")

PIB = BCB.get_table('PIB acumulado dos últimos 12 meses - Valores correntes (R$ milhões)')
PIB = PIB.astype({'data': 'datetime64[ns]'})
pib = get_var(PIB, "PIB")

IBC = BCB.get_table('Índice de Atividade Econômica do Banco Central - IBC-Br')
IBC = IBC.astype({'data': 'datetime64[ns]'})
ibc = get_var(IBC, "IBC")

ampere_indice_ONI_mensal = local.get_table("ampere_indice_ONI_mensal")
oni = get_oni(ampere_indice_ONI_mensal)

EVT = local.get_table("energia_vertida_turbinavel")
evt = get_evt(EVT)

CMO = local.get_table("cmo_semanal")
cmo_n = get_cmo(CMO, "N")
cmo_ne = get_cmo(CMO, "NE")
cmo_s = get_cmo(CMO, "S")
cmo_se = get_cmo(CMO, "SE")

proj_c_M0, proj_c_M1, proj_c_M2, proj_c_M3, proj_c_M4 = get_proj_carga("SIN")

gpr = get_GPR('GPR/data_gpr_daily_recent_modificado.csv')

earm_SIN, earm_N, earm_NE, earm_S, earm_SE = get_earm()

tna = get_var(local.get_table("ampere_indice_TNA_mensal"))

df_lin_interp = pd.concat(df_linear_interp, axis=1)

df_geral = pd.concat([pib, ibc, oni, cmo_s, cmo_se, cmo_n, cmo_ne, proj_c_M0, proj_c_M1, proj_c_M2, proj_c_M3, proj_c_M4, gpr, earm_SIN, earm_N, earm_NE, earm_S, earm_SE, tna], axis=1)

df = pd.concat([df_lin_interp, df_geral], axis=1)
df = df.dropna()
df
# -

ena
cols = [ 
 'clim',
 'clim_s30',
 'clim_s60',
 'clim_s90',
 'clim_s120',
 
 'ENA_hist-clim',
 'ENA_hist-clim_s30',
 'ENA_hist-clim_s60',
 'ENA_hist-clim_s90',
 'ENA_hist-clim_s120',
 
 'ENA_hist',
 'ENA_hist_s30',
 'ENA_hist_s60',
 'ENA_hist_s90',
 'ENA_hist_s120',
        
 'ENA_hist_MM5',      
 'ENA_hist_MM5_s30',
 'ENA_hist_MM5_s60',        
 'ENA_hist_MM5_s90',
 'ENA_hist_MM5_s120',
        
 'ENA_hist_MM7',        
 'ENA_hist_MM7_s30',
 'ENA_hist_MM7_s60',        
 'ENA_hist_MM7_s90',   
 'ENA_hist_MM7_s120',  
        
 'ENA_hist_MM15',        
 'ENA_hist_MM15_s30',
 'ENA_hist_MM15_s60',        
 'ENA_hist_MM15_s90',        
 'ENA_hist_MM15_s120',
        
 'ENA_hist_MM30',        
 'ENA_hist_MM30_s30',
 'ENA_hist_MM30_s60',        
 'ENA_hist_MM30_s90', 
 'ENA_hist_MM30_s120',
        
 'ENA_hist_MM60',
 'ENA_hist_MM60_s30',  
 'ENA_hist_MM60_s60',        
 'ENA_hist_MM60_s90',
 'ENA_hist_MM60_s120',  
        
 'ENA_hist_MM90',
 'ENA_hist_MM90_s30',
 'ENA_hist_MM90_s60',
 'ENA_hist_MM90_s90',
 'ENA_hist_MM90_s120',
]
ena = ena[cols]
ena

names = get_reversed_listofnames(df)
names
df = df[names]
df.columns

df = pd.concat([df, ena], axis=1)
df = df.dropna()
df.head()

names = get_listofnames(df)
names
df = df[names]
df.columns

# +
""" Mutual information """
import math


def make_mi_scores(X_M0, y_M0, target):
    mi_scores = mutual_info_regression(X_M0, y_M0)
    #print(pd.isnull(X_M0).any().any())
    if pd.isnull(y_M0).any().any(): 
        print((y_M0))
    if pd.isnull(X_M0).any().any(): 
        print((X_M0))
    mi_scores = pd.Series(mi_scores, name=str(target), index=X_M0.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def get_mi_scores(df, target):
    X = df.copy()
    y = X.pop(target)
    mi_scores = make_mi_scores(X, y, target)
    return mi_scores


def mi_scores_df(df):
    mi_scores = pd.DataFrame()
    for i in range(len(df.columns)-1):
        #mi_scores_temp = get_mi_scores(df[df.columns[i:]], str(df.columns[i]))
        mi_scores_temp = get_mi_scores(df[df.columns], str(df.columns[i]))
        mi_scores = pd.concat([mi_scores, mi_scores_temp], axis=1)
    #print(mi_scores)
    return mi_scores


""" Parallel processing """

from parallel_pandas import ParallelPandas
import os

ParallelPandas.initialize(n_cpu=os.cpu_count(), split_factor=16, disable_pr_bar=True)


def rolling(df, func, window):
    max_idx = df.index.max()-window
    datas = df.loc[df.index<max_idx].index.to_series()
    #print(datas)
    results = datas.p_apply(lambda a: func(df.loc[(df.index>=a) & (df.index<a+window)]))
    return results

#mi_data = rolling(df, lambda a: a, pd.Timedelta("365D"))    
#mi_scores_df(mi_data.iloc[1])


# -
mi_data = rolling(df, lambda a: mi_scores_df(a), pd.Timedelta("365D"))    
mi_data.head()

# +
x = {col: [] for col in mi_data.iloc[0]}
for d in mi_data.index:
    for n_name in mi_data.iloc[0].columns:
        x[n_name].append(mi_data.loc[d][n_name])

for n_name in mi_data.iloc[0].columns:
    CS_temp = pd.DataFrame(x[n_name], index=mi_data.index)
    CS_temp_var = CS_temp[df_lin_interp.columns]
    fig, ax = plt.subplots(figsize=(30,10))
    
    sns.heatmap(data=CS_temp_var.T, annot=False, cmap="YlGnBu", yticklabels=1,)
    ax.set(xticks=([0, 53, 104, 157, 212, 280]))
    ax.set(xticklabels = (['2019-01', '2019-07', '2020-01', '2020-07','2021-01','2021-07']))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='left')
    if not n_name.startswith("linear_interp/"):
        ax.set_title(n_name)
        name = "MI-linear_interp/var/"+n_name
    else:
        ax.set_title(n_name[13:])
        name = "MI-linear_interp/var/"+n_name[13:]
    
    plt.savefig(name, dpi=400, bbox_inches="tight")
# -
for n_name in mi_data.iloc[0].columns:
    CS_temp = pd.DataFrame(x[n_name], index=mi_data.index)
    CS_temp_ena = CS_temp[ena.columns]
    fig, ax = plt.subplots(figsize=(30,10))
    
    sns.heatmap(data=CS_temp_ena.T, annot=False, cmap="YlGnBu", yticklabels=1,)
    ax.set(xticks=([0, 53, 104, 157, 212, 280]))
    ax.set(xticklabels = (['2019-01', '2019-07', '2020-01', '2020-07','2021-01','2021-07']))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='left')
    if not n_name.startswith("linear_interp/"):
        ax.set_title(n_name)
        name = "MI-linear_interp/ena/"+n_name
    else:
        ax.set_title(n_name[13:])
        name = "MI-linear_interp/ena/"+n_name[13:]
    
    plt.savefig(name, dpi=400, bbox_inches="tight")

for n_name in mi_data.iloc[0].columns:
    CS_temp = pd.DataFrame(x[n_name], index=mi_data.index)
    CS_temp_geral = CS_temp[df_geral.columns]
    fig, ax = plt.subplots(figsize=(30,10))
    
    sns.heatmap(data=CS_temp_geral.T, annot=False, cmap="YlGnBu", yticklabels=1,)
    ax.set(xticks=([0, 53, 104, 157, 212, 280]))
    ax.set(xticklabels = (['2019-01', '2019-07', '2020-01', '2020-07','2021-01','2021-07']))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='left')
    if not n_name.startswith("linear_interp/"):
        ax.set_title(n_name)
        name = "MI-linear_interp/geral/"+n_name
    else:
        ax.set_title(n_name[13:])
        name = "MI-linear_interp/geral/"+n_name[13:]
    
    plt.savefig(name, dpi=400, bbox_inches="tight")



