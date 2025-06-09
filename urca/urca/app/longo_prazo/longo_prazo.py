from datetime import datetime
import os
from flask import render_template, request, session, redirect, current_app
from biblioteca import modulos
import json
import math
from ast import literal_eval
import pandas as pd

from app.longo_prazo import b_longo_prazo



@b_longo_prazo.route("/longo_prazo", methods=["GET"])
def longo_prazo():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    output_folder = os.path.join('web', 'static', 'tasks_saida', 'longo_prazo', 'M+0', 'output')
    
    # obter os arquivos iniciados em markov-saida_dash
    files = [f for f in os.listdir(output_folder) if f.startswith("longo_prazo_saida_")]
    
    # obter as datas dos arquivos (em XX_XX_XXXX) e ordene para obter o mais atual
    datas = [f.replace('longo_prazo_saida_', '').replace("_salto", "").split(".")[0] for f in files] 
    
     # transformar em datetime
    datas = [datetime.strptime(data, "%d_%m_%Y") for data in datas]
    
    
    # remover datas duplicadas
    datas = list(set(datas))
    
    
    # organizar da data mais recente à mais antiga
    datas.sort(reverse=True)
    
    # transformar em string novamente
    datas = [data.strftime("%d_%m_%Y") for data in datas]
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("longo_prazo/longo_prazo.html", usuario=session["usuario"], menu=menu, datas=datas)
