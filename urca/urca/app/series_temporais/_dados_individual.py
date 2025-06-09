from flask import request, session, redirect, current_app
from biblioteca import modulos
import json
import math
from ast import literal_eval
import pandas as pd

from app.series_temporais import b_series_temporais



@b_series_temporais.route("/_obter_precos", methods=["POST"])
def _obter_precos():
    form = request.form
    visualizacao = form.get("visualizacao")
    dados = current_app.obter_dados_historicos()
    preco, preco_h, preco_r = dados["preco"], dados["preco_h"], dados["preco_r"]
    pld, pld_piso_teto, PLD = dados["pld"], dados["pld_piso_teto"], dados["PLD"]
    mes = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN','JUL', 'AGO', 'SET', 'OUT', 'NOV', 'DEZ']
    
    produtos = dados['produtos']
    
    if visualizacao == "Produto Individual":
        print(f"Produtos: {produtos}")
        produto_selecionado = form.get("produto")
        if not produto_selecionado:
            produto_selecionado = produtos[0]
            
        tabela_preco_h = preco_h[preco_h["produto"] == produto_selecionado]
        
        
        data = {
            "preco": tabela_preco_h.to_dict('index'),
            "produtos": produtos.tolist(),
            #"mes": mes,
            "pld": pld.to_dict('index'),
            #"pld_piso_teto": pld_piso_teto.to_dict('index'),
            #"PLD": PLD.to_dict('index'),
        }        
        return json.dumps(data, default=str)
    
    if visualizacao == "Grupo de Produtos":
        grupo_selecionado = form.get("grupo_selecionado")
        if not grupo_selecionado:
            grupo_selecionado = mes[0]
            
        produtos_grupo = preco_h[preco_h['produto'].str.contains(grupo_selecionado, case=False)]['produto'].unique()
        
        print(f"Produtos Grupo: {produtos_grupo}")
        produtos_dict = {index: produto for index, produto in enumerate(produtos_grupo)}
        
        print(f"Produtos Dict: {produtos_dict}")

        data = {
            "preco": preco_h[preco_h['produto'].isin(produtos_grupo)].to_dict('index'),
            "produtos": produtos_dict,
            "mes": mes,
            "pld": pld.to_dict('index'),
            "pld_piso_teto": pld_piso_teto.to_dict('index'),
        }
        
        return json.dumps(data, default=str)
    
    if visualizacao == "Rollof":
        submercados = preco_r.submercado.unique()
        submercado = form.get("submercado")
        if not submercado:
            submercado = submercados[0]
            
        maturacao = form.get("maturacao")
        if not maturacao:
            data = {
                #"preco": preco_r.to_dict('index'),
                "submercados": submercados.tolist(),
                "mes": mes,
            }
        
            return json.dumps(data, default=str)
        maturacao = int(form.get("maturacao"))
        print(f"Submercado: {submercado}, Maturacao: {maturacao}, Dados: {preco_r}")
        dados = preco_r[preco_r['submercado'].str.contains(submercado, case=False)]
        
        print(f"Submercado: {submercado}, Maturacao: {maturacao}, Dados: {dados}")
        
        dados = dados[dados['M'] == maturacao]
        print(f"Submercado: {submercado}, Maturacao: {maturacao}, Dados: {dados}")
        
        data = {
            "dados": dados.to_dict('index'),
            "submercados": submercados.tolist(),
            "mes": mes,
        }
        
        print(f"Data: {data}")
        
        return json.dumps(data, default=str)
    
    data = {
        "preco": preco.to_dict(orient='records'),
        "preco_h": preco_h.to_dict(orient='records'),
        "preco_r": preco_r.to_dict(orient='records'),
        "mes": mes,
        "produtos": produtos.tolist(),
        "pld": pld.to_dict(orient='records'),
        "pld_piso_teto": pld_piso_teto.to_dict(orient='records'),
        "PLD": PLD.to_dict(orient='records'),
    }
    return json.dumps(data, default=str)