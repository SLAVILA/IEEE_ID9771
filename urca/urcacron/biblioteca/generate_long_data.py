#%%
from bcb import Expectativas
import pandas as pd

try:
    from .db_api import Local, UrcaDb, BcbDb, AmpereDB
except ImportError:
    from db_api import Local, UrcaDb, BcbDb, AmpereDB

import os

#ena_atual
def generate_ena_atual(filename):
    ena_atual = Local().get_table("ampere_ENA_hist_REE_new").set_index("data")
    col = list(ena_atual.columns)
    col.sort()
    ena_atual = ena_atual[col]
    ena_atual.to_csv(filename)
    return ena_atual

#%%
#ena_prevista
def generate_ena_prevista(filename):
    ena = Local().get_table("ampere_ena_prevista_rees")
    ena = ena[ena.cod_rees == 1] # sudeste


    reforecast = ena[ena.modelo == "CFSV2-REFORECAST"]

    informes = ena[ena.modelo == "CFSV2-INFORMES"]
    informes = informes[informes.data > reforecast.data.max()]
    
    all_ena = pd.concat([reforecast, informes])
    #print(reforecast)
    #print(informes)
    ena_prevista = all_ena.drop(columns=["cod_rees", "modelo", "nome_rees"])
    ena_prevista.to_csv(filename, index=False, date_format="%m/%d/%Y", header=False)
    return ena_prevista

#%%
# historico_carga;
def generate_historico_carga(filename):
    ipdo = UrcaDb().get_table("ipdo")
    ipdo.data = pd.to_datetime(ipdo.data)

    carga_ipdo = ipdo[["regiao", "data", "carga_realizada"]].rename(columns={"regiao": "id_subsistema", "carga_realizada": "valor"})
    carga_ipdo = carga_ipdo.set_index(["data", "id_subsistema"])
    carga_ipdo = carga_ipdo[carga_ipdo.valor != 0]

    carga_old = UrcaDb().get_table("historico_carga")
    carga_old.data = pd.to_datetime(carga_old.data)
    carga_old = carga_old.set_index(["data", "id_subsistema"])
    carga_old = carga_old[carga_old.valor != 0]

    carga = pd.concat([carga_old.drop(index=(carga_ipdo.index.intersection(carga_old.index))), carga_ipdo])
    carga = carga.reset_index()

    sub = UrcaDb().get_table("subsistema").rename(columns={"nome": "subsistema"})
    carga = carga.join(sub.set_index("id_subsistema"), on="id_subsistema")
    carga = carga[["id_subsistema", "idhistorico_carga", "subsistema", "data", "valor"]]
    carga.to_csv(filename, date_format="%d/%m/%Y", index=False)
    return carga

#%%
# carga_prevista.xlsx
def generate_carga_prevista(filename):
    proj_carga = Local().get_table("projecao_carga_raw")
    proj_carga.to_csv(filename, index=False)
    return proj_carga

#%%
# Piso_teto_PLD.xlsx
def generate_piso_teto_PLD(filename):
    piso_teto_pld = Local().get_table("piso_teto_pld")
    piso_teto_pld.to_csv(filename, index=False, date_format="%Y")
    return piso_teto_pld

#%%
# historico_ipca.csv
def generate_historico_ipca(filename):
    ipca = BcbDb().get_table("Índice nacional de preços ao consumidor - amplo (IPCA)")
    ipca.to_csv(filename, date_format="%d/%m/%Y")
    return ipca

#%%
# historico_ipca.csv
def generate_expectativa_ipca(filename):
    def months_diff(row):
        m = row.Data.month
        y = row.Data.year
        ref_m, ref_y = row.DataReferencia.split("/")
        return int(ref_m)-m + (int(ref_y)-y)*12
    
    em = Expectativas()
    ep = em.get_endpoint("ExpectativaMercadoMensais")
    q = ep.query()
    q = q.filter(ep.Indicador == 'IPCA')
    q = q.filter(ep.baseCalculo == 1)
    q = q.select(ep.Data, ep.DataReferencia, ep.Mediana)
    a = q.collect()
    
    IPCA_exp = a
    IPCA_exp.Data = pd.to_datetime(IPCA_exp.Data, format="%Y-%m-%d")
    IPCA_exp["Month"] = IPCA_exp.apply(months_diff, axis=1)
    IPCA_exp = IPCA_exp[IPCA_exp["Month"] >= 0]
    IPCA_exp
    
    def process(df):
        data = pd.DataFrame(index=range(0, 50), columns=["Mediana"])
        data.update(df.set_index("Month"))
        data.Mediana = data.Mediana.ffill()
        dataT = data.T
        dataT.index = [df.Data.iloc[0]]
        return dataT
        
    b = IPCA_exp.groupby("Data").apply(process)
    b = b.droplevel(level=1)
    b.columns = [f"M+{i}" for i in b.columns]
    b.to_csv(filename)
    return b

def generate_all(path="app/longo_prazo/dados"):
    os.makedirs(path, exist_ok=True)

    AmpereDB().get_table("ampere_ena_hist_rees", update=True)

    generate_ena_atual(os.path.join(path, "ena_atual.csv"))
    generate_ena_prevista(os.path.join(path, "ena_prevista.csv"))
    generate_historico_carga(os.path.join(path, "historico_carga.csv"))
    generate_carga_prevista(os.path.join(path, "carga_prevista.csv"))
    generate_piso_teto_PLD(os.path.join(path, "Piso_teto_PLD.csv"))
    generate_historico_ipca(os.path.join(path, "historico_ipca.csv"))
    generate_expectativa_ipca(os.path.join(path, "expectativa_ipca.csv"))

if __name__ == '__main__':
    generate_all("../app/longo_prazo/dados")
    #preco = UrcaDb().get_table("preco_bbce")
    #preco.to_csv("bbce_data_raw.csv")