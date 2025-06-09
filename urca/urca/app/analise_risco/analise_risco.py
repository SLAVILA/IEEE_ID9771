from datetime import datetime
import os
from flask import render_template, request, session, redirect, current_app
from biblioteca import modulos
import json
import math
from ast import literal_eval
import pandas as pd

from app.analise_risco import b_analise_risco


@b_analise_risco.route("/analise_risco_curto_prazo", methods=["GET"])
def analise_risco_curto():
    # Verifica se o usuário está logado
    if "usuario" not in session.keys():
        return redirect("login")
    
    try:
        output_dir = os.path.join('web', 'static', 'tasks_saida', 'estocastico', 'M+0', 'risco')
    
        arquivos = os.listdir(output_dir)
    
        # remover estocastico_ do arquivo
        arquivos = [arquivo.replace('estocastico_', '').replace('.json', '') for arquivo in arquivos]
        
        # obter as datas dos arquivos (em XX_XX_XXXX) e ordene para obter o mais atual
        datas = [f.replace('estocastico_', '').split(".")[0] for f in arquivos] 
    
         # transformar em datetime
        datas = [datetime.strptime(data, "%d_%m_%Y") for data in datas]
    
    
        # remover datas duplicadas
        datas = list(set(datas))
    
    
        # organizar da data mais recente à mais antiga
        datas.sort(reverse=True)
    
        # transformar em string novamente
        datas = [data.strftime("%d_%m_%Y") for data in datas]
    except:
        arquivos = []
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("analise_risco/analise_risco_curto.html", usuario=session["usuario"], menu=menu, array_arquivos=datas)

@b_analise_risco.route("/analise_risco_longo_prazo", methods=["GET"])
def analise_risco_longo_prazo():
    # Verifica se o usuário está logado
    if "usuario" not in session.keys():
        return redirect("login")
    
    output_folder = os.path.join('web', 'static', 'tasks_saida', 'longo_prazo', 'M+0', 'risco')
    
    # obter os arquivos iniciados em markov-saida_dash
    files = [f for f in os.listdir(output_folder) if f.startswith("longo_prazo_") and (f.endswith("salto.json") or not f.endswith("_.json"))]
    
    # obter as datas dos arquivos (em XX_XX_XXXX) e ordene para obter o mais atual
    datas = [f.replace('longo_prazo_', '').replace("_salto", "").split(".")[0] for f in files] 
    
     # transformar em datetime
    datas = [datetime.strptime(data, "%d_%m_%Y") for data in datas]
    
    
    # remover datas duplicadas
    datas = list(set(datas))
    
    
    # organizar da data mais recente à mais antiga
    datas.sort(reverse=True)
    
    # transformar em string novamente
    datas = [data.strftime("%d_%m_%Y") for data in datas]
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("analise_risco/analise_risco_longo.html", usuario=session["usuario"], menu=menu, array_arquivos=datas)