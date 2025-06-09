from flask import current_app, jsonify, request, session, redirect
from biblioteca import modulos
import json
import math
from ast import literal_eval
from plotly.io import to_json

from app.series_temporais import b_series_temporais
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import os

from biblioteca.db_api import BcbDb
from biblioteca.db_api import UrcaDb
from biblioteca.db_api import Local
BCB = BcbDb()
local = Local()
urca = UrcaDb()

# +
min = datetime.datetime(2021, 7, 1)
max = datetime.datetime(2023, 3, 30)

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

    #mask = (VAR['data'] >= min) & (VAR['data'] <= max)
    df_temp = VAR#.loc[mask]
    df_temp = df_temp.set_index('data')
    if "valor" in df_temp.columns:
        df_temp = df_temp.rename(columns={'valor': variavel})
    return df_temp


def get_GPR(path):
    df_temp = pd.read_csv(path, parse_dates=['date'], dayfirst=True, decimal=',').rename(columns={"date" : "data"})
    df_temp = df_temp.set_index('data')
    return df_temp


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
    earm = pd.merge(current_app.dados_historicos['subsistema'].rename(columns={"nome": "nome_regiao","id_subsistema":"regiao"}), current_app.dados_historicos['historico_hidrologia'], on="regiao")
    earm = earm[['nome_regiao', 'data','earm']].astype({'data' : 'datetime64[ns]'})
    #earm = earm[earm['data'].dt.year >= 2014]
    #earm = earm[earm['data'] >= min]
    #earm = earm[earm['data'] <= max]

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
    cmo_temp = cmo_temp.dropna()
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

    #mask = (oni['data'] >= min) & (oni['data'] <= max)
    df_temp = oni#.loc[mask]
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
    #df_shifted_s3 = df.shift(periods=period*3, freq=freq).add_suffix("_s90")
    #df_shifted_s4 = df.shift(periods=period*4, freq=freq).add_suffix("_s120")
    df_temp = pd.concat([df, df_shifted_s1, df_shifted_s2], axis=1)
    #df_temp = pd.concat([df, df_shifted_s1, df_shifted_s2, df_shifted_s3, df_shifted_s4], axis=1)
    return df_temp


def get_ENA(ena_path):
    df_temp = pd.read_csv(ena_path).rename(columns={"hist" : "ENA_hist", "diff" : "ENA_hist-clim"}).astype({'data' : 'datetime64[ns]'})
    mask = (df_temp['data'] >= min) & (df_temp['data'] <= max)
    df_temp = df_temp.loc[mask]
    df_temp = df_temp.set_index('data')
    #df_temp = MM_df(df_temp, 5)
    df_temp = shift_df(df_temp, 30, "D")
    #df_temp = df_temp.dropna()
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
import plotly.express as px
import plotly.graph_objects as go

# Funções para carregar dados com cache utilizando variáveis globais
cache_ONIm_data = None
cache_TNA_data = None
cache_ProjC_data = None
cache_CMO_data = None
cache_EARm_data = None
cache_Clim_data = None
cache_IPCA_data = None
cache_preco_bbce_data = None
cache_TPPLD_data = None
cache_enaCFSV2_data = None

# Funções para carregar dados sem streamlit
def load_ONIm_data():
    global cache_ONIm_data
    if cache_ONIm_data is None or passou_tempo():
        ampere_indice_ONI_mensal = local.get_table("ampere_indice_ONI_mensal")
        cache_ONIm_data = get_oni(ampere_indice_ONI_mensal)
    return cache_ONIm_data

def load_TNA_data():
    global cache_TNA_data
    if cache_TNA_data is None or passou_tempo():
        cache_TNA_data = get_var(local.get_table("ampere_indice_TNA_mensal"))
    return cache_TNA_data

def load_ProjC_data():
    global cache_ProjC_data
    if cache_ProjC_data is None or passou_tempo():
        cache_ProjC_data = get_proj_carga("SIN")
    return cache_ProjC_data

def load_CMO_data():
    global cache_CMO_data
    if cache_CMO_data is None or passou_tempo():
        CMO = local.get_table("cmo_semanal")
        cache_CMO_data = (get_cmo(CMO, "N"), get_cmo(CMO, "NE"))
    return cache_CMO_data

def load_EARm_data():
    global cache_EARm_data
    if cache_EARm_data is None or passou_tempo():
        earm_SIN, earm_N, earm_NE, earm_S, earm_SE = get_earm()
        cache_EARm_data = earm_SIN
    return cache_EARm_data

def load_Clim_data():
    global cache_Clim_data
    if cache_Clim_data is None or passou_tempo():
        ena = get_ENA(os.path.join("app", "series_temporais", "ENA_SIN.csv"))
        cols = ['clim_s30', 'clim_s60']
        cache_Clim_data = ena[cols]
    return cache_Clim_data

def load_IPCA_data():
    global cache_IPCA_data
    if cache_IPCA_data is None or passou_tempo():
        IPCA = BCB.get_table("Índice nacional de preços ao consumidor - amplo (IPCA)")
        IPCA["data"] = pd.to_datetime(IPCA["data"], format="%d/%m/%Y")
        IPCA = IPCA.set_index('data')
        IPCA.valor = IPCA.valor.astype("float") # editado
        cache_IPCA_data = IPCA[IPCA.index > "2000-01-01"]
    return cache_IPCA_data

def load_preco_bbce_data():
    global cache_preco_bbce_data
    if cache_preco_bbce_data is None or passou_tempo():
        #preco_bbce = urca.get_table("preco_bbce")
        preco_bbce = current_app.dados_historicos['precobbce']
        preco_bbce = preco_bbce.sort_values(by='data_hora').set_index('data_hora')
        preco_bbce = preco_bbce[preco_bbce.tipo == 1]
        cache_preco_bbce_data = preco_bbce['preco']
    return cache_preco_bbce_data

def load_TPPLD_data():
    global cache_TPPLD_data
    if cache_TPPLD_data is None or passou_tempo():
        cache_TPPLD_data = local.get_table("piso_teto_pld")
    return cache_TPPLD_data

def load_enaCFSV2_data():
    global cache_enaCFSV2_data
    if cache_enaCFSV2_data is None or passou_tempo():
        ampere_ena_prevista_rees = local.get_table("ampere_ena_prevista_rees")
        cfsv2 = ampere_ena_prevista_rees[ampere_ena_prevista_rees['modelo'] == "CFSV2-INFORMES"]
        print(cfsv2)
        rees = cfsv2['nome_rees'].unique()
        cache_enaCFSV2_data = (cfsv2, rees)
    return cache_enaCFSV2_data

def obter_dados_grafico(tipo):
    if tipo == "ONIm":
        return load_ONIm_data(), None
    elif tipo == "TNA":
        return load_TNA_data(), None
    elif tipo == "Proj_C":
        return load_ProjC_data(), None
    elif tipo == "CMO":
        return load_CMO_data(), None
    elif tipo == "EARm":
        return load_EARm_data(), None
    elif tipo == "Clim":
        return load_Clim_data(), None
    elif tipo == "IPCA":
        return load_IPCA_data(), None
    elif tipo == "Preços BBCE":
        return load_preco_bbce_data(), None
    elif tipo == "Teto e Piso PLD":
        return load_TPPLD_data(), None
    elif tipo == "ENA prevista CFSV2":
        return load_enaCFSV2_data()

def gerar_grafico_plot(tipo, data_escolhida=None, rees_escolhido=None):
    """
    Função genérica para gerar gráficos com base no tipo de gráfico solicitado.
    """
    fig = None
    
    dados, rees = obter_dados_grafico(tipo)
    
    # Gráfico de ONIm
    if tipo == "ONIm":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dados.index,  # X-axis data (dates)
            y=dados.iloc[:, 0],  # Y-axis data (the first column of the data)
            mode='lines',  # Line mode
            name='ONIm'  # Name for the legend
        ))
        fig.update_layout(
            title="Gráfico de ONIm",
            xaxis_title="Data",
            yaxis_title="Valores",
            legend_title="Legenda"
        )

    # Gráfico de TNA
    elif tipo == "TNA":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados.iloc[:, 0],  # Assuming the first column contains the TNA data
            mode='lines',
            name='TNA'
        ))
        fig.update_layout(
            title="Gráfico de TNA",
            xaxis_title="Data",
            yaxis_title="Valores",
            legend_title="Legenda"
        )

    # Gráfico de Clim
    elif tipo == "Clim":
        clim_s30 = dados['clim_s30']
        clim_s60 = dados['clim_s60']
        indice = dados.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indice, y=clim_s30, mode='lines', name='shift de 30 dias'))
        fig.add_trace(go.Scatter(x=indice, y=clim_s60, mode='lines', name='shift de 60 dias'))
        fig.update_layout(
            title="Climatologia de ENA",
            xaxis_title="Datas",
            yaxis_title="Climatologia de ENA",
            legend_title="Legenda"
        )

    # Gráfico de Projecao de carga
    elif tipo == "Proj_C":
        proj_c_M0, proj_c_M1, proj_c_M2, proj_c_M3, proj_c_M4 = dados
        indice = proj_c_M0.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indice, y=proj_c_M0["Proj_c_M0"], mode='lines', name='M+0'))
        fig.add_trace(go.Scatter(x=indice, y=proj_c_M1["Proj_c_M1"], mode='lines', name='M+1'))
        fig.add_trace(go.Scatter(x=indice, y=proj_c_M2["Proj_c_M2"], mode='lines', name='M+2'))
        fig.add_trace(go.Scatter(x=indice, y=proj_c_M3["Proj_c_M3"], mode='lines', name='M+3'))
        fig.add_trace(go.Scatter(x=indice, y=proj_c_M4["Proj_c_M4"], mode='lines', name='M+4'))
        fig.update_layout(
            title="Projeção de Carga",
            xaxis_title="Índice",
            yaxis_title="Projeção de Carga",
            legend_title="Projeções"
        )

    # Gráfico de CMO
    elif tipo == "CMO":
        cmo_n, cmo_ne = dados
        indice_cmo = cmo_n.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=indice_cmo, y=cmo_n["CMO_N"], mode='lines', name='CMO N'))
        fig.add_trace(go.Scatter(x=indice_cmo, y=cmo_ne["CMO_NE"], mode='lines', name='CMO NE'))
        fig.update_layout(
            title="CMO",
            xaxis_title="Índice",
            yaxis_title="CMO",
            legend_title="Legenda"
        )

    # Gráfico de EARm
    elif tipo == "EARm":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados.iloc[:, 0],  # Assuming the first column contains the EARm data
            mode='lines',
            name='EARm'
        ))
        fig.update_layout(
            title="Gráfico de EARm (SIN)",
            xaxis_title="Data",
            yaxis_title="Valores",
            legend_title="Legenda"
        )

    # Gráfico de IPCA
    elif tipo == "IPCA":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dados.index,
            y=dados.iloc[:, 0],  # Assuming the first column contains the IPCA data
            mode='lines',
            name='IPCA'
        ))
        fig.update_layout(
            title="Gráfico de IPCA",
            xaxis_title="Data",
            yaxis_title="Valores",
            legend_title="Legenda"
        )

    elif tipo == "Preços BBCE":
        print(dados)  # Para verificar a estrutura dos dados

        # Verifique se 'dados' é uma Série
        if isinstance(dados, pd.Series):
            x = dados.index  # Use o índice como x
            y = dados.values  # Use os valores da Série como y
        else:
            raise TypeError("Tipo de dados inesperado. Esperado uma Série.")

        # Crie o gráfico usando Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Preço BBCE'
        ))
        fig.update_layout(
            title="Gráfico de preço BBCE",
            xaxis_title="Data",
            yaxis_title="Valores",
            legend_title="Legenda"
        )


    # Gráfico de Teto e Piso PLD
    elif tipo == "Teto e Piso PLD":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dados["Ano"], y=dados["PLD_minimo"], mode='lines', name='Piso'))
        fig.add_trace(go.Scatter(x=dados["Ano"], y=dados["PLD_maximo"], mode='lines', name='Teto'))
        fig.update_layout(
            title="TPPLD",
            xaxis_title="Data",
            yaxis_title="PLD",
            legend_title="Legenda"
        )

    # Gráfico de ENA prevista CFSV2
    elif tipo == "ENA prevista CFSV2":
        cfsv2 = dados
        df_ree = cfsv2[cfsv2['nome_rees'] == rees_escolhido].drop(columns=["cod_rees", "modelo", "nome_rees"])
        data = []
        for i in range(len(df_ree)):
            index = [df_ree.iloc[i]['data'] + datetime.timedelta(days=x) for x in range(311)]
            df_temp = df_ree.iloc[i]
            df_temp.index = [df_temp.index[0]] + index[1:]
            df = pd.DataFrame(data=df_temp, index=index)
            df = df.iloc[1:]
            data.append(df)
            if str(data[i].index[0]) == str(data_escolhida):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data[i].index,
                    y=data[i].iloc[:, 0],  # Assuming the first column contains the ENA data
                    mode='lines',
                    name='ENA prevista'
                ))
                fig.update_layout(
                    title="Gráfico de ENA",
                    xaxis_title="Data",
                    yaxis_title="ENA",
                    legend_title="Legenda"
                )
                break

    return fig

def passou_tempo():
    ultimo_update = current_app.dados_historicos['ultimo_update']
    
    # verificar sejá passou mais de 1 hora
    if (datetime.datetime.now() - ultimo_update).seconds > (3600 * 6):
        current_app.dados_historicos['ultimo_update'] = datetime.datetime.now()
        return True
    return False

datas = None
def obter_datas():
    global datas
    if not datas:
        cfsv2, rees = load_enaCFSV2_data()
        datas = cfsv2['data'].unique()[2:]
    
        # transformar em lista de strings
        datas = [str(data) for data in datas]
    rees = load_enaCFSV2_data()[1].tolist()
    return datas, rees

@b_series_temporais.route("/_obter_datas", methods=["POST"])
def _obter_datas():
    datas, rees = obter_datas()
    return jsonify({'datas': datas, 'rees': rees}), 200
    
@b_series_temporais.route("/_gerar_grafico", methods=["POST"])
def gerar_grafico():
    """
    Endpoint para gerar gráficos com base nos parâmetros enviados pelo frontend.

    1. Verifica se o usuário está logado
    2. Valida as permissões do usuário
    3. Recebe o tipo de gráfico e data, se necessário
    4. Gera o gráfico correspondente com os dados apropriados
    5. Retorna o gráfico gerado para o frontend

    Returns:
        Retorna: Gráfico gerado
    """
    # Verifica se o usuário está logado
    if "usuario" not in session.keys():
        return redirect("login")
    
    # Verifica permissões do usuário
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    # Recebe os dados do frontend (gráfico a ser gerado e data, se necessário)
    data = request.form.get("data", None)  # Recebe a data, se fornecida
    grafico_tipo = request.form.get("grafico", None)  # Tipo de gráfico a ser gerado
    rees_escolhido = request.form.get("rees", None)  # Recebe o ree, se fornecido
    
    if not grafico_tipo:
        return jsonify({"error": "Tipo de gráfico não fornecido"}), 400

    # Gera o gráfico correspondente
    fig = gerar_grafico_plot(grafico_tipo, data, rees_escolhido)
    fig = fig.to_json()
    
    return jsonify({
        "plot": fig
    }), 200


@b_series_temporais.route("/_series_temporais", methods=["POST"])
def _series_temporais():
    """
    Rota para obter os dados da lista 

    1. Verifica se o usuário está logado
    2. Obtem as permissoes do usuario, e retorna caso não possua permissão
    3. Obtem os dados do formulario
    4. Santiza os dados

    Returns:
        Retorna: Sucesso
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtem as permissoes do usuario, e retorna caso não possua permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])

    
    f = request.form

    page = 1
    limit = 10
    contagem_total = 0
    total_pages = 0
    sort_field = ""
    sort_ord = "ASC"
    search_query = ""

    if "pagination[page]" in f.keys():
        page = int(f.get("pagination[page]"))

    if "pagination[perpage]" in f.keys():
        limit = int(f.get("pagination[perpage]"))

    if "sort[field]" in f.keys():
        sort_field = f.get("sort[field]")

    if "sort[sort]" in f.keys():
        sort_ord = f.get("sort[sort]")

    if "query[generalSearch]" in f.keys():
        search_query = f.get("query[generalSearch]")
    
    contagem = modulos.Banco().sql(
        "SELECT \
            COUNT(id_historico_hidrologia) as count \
        FROM urca.historico_hidrologia",
        (),
    )
    

    contagem_total = contagem[0]["count"]
    total_pages = math.floor(contagem_total / limit)

    # Calculate the starting and ending row numbers for the given page
    start_row = (page - 1) * limit + 1
    end_row = page * limit

    # Execute the SQL query with pagination
    resposta = modulos.Banco().sql(
        "SELECT * \
        FROM \
            (SELECT \
                *, \
                ROW_NUMBER() OVER (ORDER BY {} {}) AS RowNum \
            FROM urca.historico_hidrologia) AS SubQuery \
        WHERE \
            RowNum BETWEEN %s AND %s".format(sort_field, sort_ord.upper()),
        (start_row, end_row)
    )

    
    print(f"Resposta: {resposta}")
    
    for dado in resposta:
        dado['data'] = dado['data'].strftime('%Y-%m-%d')


    resultado = {
        "meta": {
            "page": page,
            "pages": total_pages,
            "perpage": limit,
            "total": contagem_total,
            "sort": "asc",
            "field": "id",
        },
        "data": resposta,
    }

    return json.dumps(resultado)
