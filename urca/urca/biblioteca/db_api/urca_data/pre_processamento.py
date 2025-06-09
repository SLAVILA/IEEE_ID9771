from .utils import get_maturidade
from .process_produtos_bbce import remove_hifen, p5_corrections
import pandas as pd
import numpy as np
import datetime
import time
import os

def preco_bbce(preco, raw=False, **kwargs):
    # outlier encontrado, acreditamos que o nome deste produto esteja faltando o "MEN"
    preco['data'] = preco['data_hora'].dt.date
    preco['data'] = preco['data'].astype("datetime64[ns]")
    preco = preco.reindex(columns=['id_bbce', 'produto', 'data_hora', 'data', 'volume', 'preco', 'tipo']).set_index('id_bbce')
    if not raw:
        preco = remove_hifen(preco)
        preco = preco.replace(p5_corrections(preco))
    
    return preco

def preco_bbce_men_preco_fixo(preco, path, outliers=None, **kwargs):
    # pega apenas os produtos de Preço Fixo
    preco = preco[preco.tipo == 1].drop(["tipo"], axis=1)
    
    if outliers:
        outliers = pd.read_csv(os.path.join(path, outliers), index_col=0)
        preco = preco.drop(index=outliers.index)
    
    # padroniza nomes dos produtos e filtra apenas produtos mensais de fonte não incentivada
    validos = [] # produtos mensais
    rename = {}  # produtos que tem que remover " - Preço Fixo" do final
    for prod in preco.produto.unique():
        partes = prod.strip().replace("- ", "").replace("_", " ").replace("  ", " ").split(" ")
        if "CON" != partes[1]:
            # não é de fonte não incentivada
            continue
        if "MEN" != partes[2]:
            # não é mensal
            continue

        validos.append(prod)
        if prod.endswith(" - Preço Fixo"):
            rename[prod] = prod[:-len(" - Preço Fixo")]

    # seleciona produtos mensais de fonte não incentivada
    linhas_filtradas = preco["produto"].apply(lambda prod: prod in validos)
    preco = preco[linhas_filtradas]
    
    # renomeia os produtos
    preco["produto"] = preco["produto"].apply(lambda x: rename[x] if x in rename else x)
    
    # filtra datas que são inferiores à realização do produto
    next_month = {
        "jan": 2, "fev":  3, "mar":  4, "abr":  5,
        "mai": 6, "jun":  7, "jul":  8, "ago":  9,
        "set": 10, "out": 11, "nov": 12, "dez":  1
    }
    
    expiracoes = {}
    for prod in preco.produto.unique():
        # example: JUL/18
        month, year = prod.split(" ")[-1].split("/")
        y = 2000+int(year)
        m = next_month[month.lower()]
        if m == 1:
            # era dezembro, logo é no próximo ano
            y += 1
        expiracoes[prod] = datetime.date(y, m, 1)
    preco["expiracao"] = preco["produto"].apply(lambda x: expiracoes[x]).astype("datetime64[ns]")
    preco["submercado"] = preco["produto"].apply(lambda x: x[:2])

    preco = preco[preco["data"] < preco["expiracao"]]

    preco = preco[preco.data < preco.expiracao]
    return preco.reindex(columns=['produto', 'expiracao', 'submercado', 'data', 'data_hora', 'volume', 'preco'])

def preco_bbce_men_preco_fixo_diario(preco, **kwargs):
    preco = preco.drop(preco[preco.volume == 0].index)
    
    grouped = preco.groupby(["produto", 'submercado', "expiracao", "data"])[["preco", "volume"]]
    VWAP = grouped.apply(lambda x: np.average(x["preco"], weights=x["volume"]))
    
    #df = preco.groupby(["produto", "expiracao", "data"])["volume"].sum()
    df = grouped.sum()
    df["VWAP"] = VWAP.values
    del df["preco"]
    return get_maturidade(df.reset_index(), "produto", "data").reset_index(drop=True)

