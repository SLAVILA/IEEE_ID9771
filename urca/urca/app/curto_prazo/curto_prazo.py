from datetime import datetime
import os
from flask import render_template, request, session, redirect, current_app
from biblioteca import modulos
import json
import math
from ast import literal_eval
import pandas as pd
from ._redes_neurais import TesteURCA

from app.curto_prazo import b_curto_prazo



@b_curto_prazo.route("/previsoes_curto_prazo_analise_tecnica", methods=["GET"])
def analise_tecnica():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("curto_prazo/curto_prazo_analise_tecnica.html", usuario=session["usuario"], menu=menu)



@b_curto_prazo.route("/previsoes_curto_prazo_markov", methods=["GET"])
def markov():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    output_dir = os.path.join('app', 'curto_prazo', 'dados_markov')
    
    
    
    # obter diretorios
    try:
        arquivos = os.listdir(output_dir)
    
        # remover extensão do arquivo
        arquivos_totais = [os.path.splitext(arquivo)[0] for arquivo in arquivos]
    
        output_dir = os.path.join('web', 'static', 'tasks_saida', 'markov')
        
        arquivos = [diretorio for diretorio in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, diretorio))]
        
        for diretorio in arquivos:
            # obter os arquivos iniciados em markov-saida_dash
            files = [f for f in os.listdir(os.path.join(output_dir, diretorio)) if f.startswith("markov-saida_") and f.endswith(".json")]
    
    
            # obter as datas dos arquivos (em XX_XX_XXXX) e ordene para obter o mais atual
            datas = [f.replace("markov-saida_", "").split(".")[0] for f in files if f.endswith(".json")]
            
            # reverse
            datas = sorted(datas, key=lambda x: datetime.strptime(x, '%d_%m_%Y'), reverse=True)
            
            print(datas)
            
    except Exception as e:
        print(e)
        arquivos = []
        arquivos_totais = []
        datas = []
    
    novos_arquivos = []
    for arquivo in arquivos:
        novos_arquivos.append(arquivo.replace("rolloff suavizado ", "").replace("SE -> VWAP", ""))
        
    # novos_arquivos.reverse()
        
    print(novos_arquivos)
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("curto_prazo/curto_prazo_markov.html", usuario=session["usuario"], menu=menu, arquivos=arquivos, arquivos_totais=arquivos_totais, datas=datas, novos_arquivos=novos_arquivos)



@b_curto_prazo.route("/previsoes_curto_prazo_estocastico", methods=["GET"])
def estocastico():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    try:
        output_dir = os.path.join('web', 'static', 'tasks_saida', 'estocastico', 'M+0', 'output')
    
        arquivos = os.listdir(output_dir)
    
    
    
        # obter as datas dos arquivos (em XX_XX_XXXX) e ordene para obter o mais atual
        datas = [f.replace("estocastico_saida_", "").split(".")[0] for f in arquivos if f.endswith(".json")]
            
        # reverse
        datas = sorted(datas, key=lambda x: datetime.strptime(x, '%d_%m_%Y'), reverse=True)
    except Exception as e:
        print(e)
        arquivos = []
        
    try:
        output_dir = os.path.join('web', 'static', 'tasks_saida', 'estocastico', 'M+0')
    
        arquivos_auxiliares = os.listdir(output_dir)
        
        
    except:
        arquivos_auxiliares = []
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("curto_prazo/curto_prazo_estocastico.html", usuario=session["usuario"], menu=menu, array_arquivos=datas, arquivos_auxiliares=arquivos_auxiliares)



@b_curto_prazo.route("/previsoes_curto_prazo_redes_neurais", methods=["GET"])
def rede_neural():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    output_dir = os.path.join('web', 'static', 'tasks_saida', 'rede_neural', str(session['usuario']['id']), 'saidas_univariado')
    
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, 'dados_preco', 'linear interpol', 'VWAP pentada', 'rolloff suavizado M+1 SE -> VWAP.csv')

    # Carrega o arquivo CSV
    VWAP = pd.read_csv(csv_path, index_col=1, parse_dates=True)
    
    path_best = os.path.join(script_dir, 'dados_rede_neural', "diff_fuzzy_UNI_autoregressive_suavizado", "teste_exp_block_diff_VWAP_064_units_1_layers_vanilla")
    test = TesteURCA(path_best, load=True)

    # Número de passos de entrada e saída
    inp_steps = test._kwargs['model_kwargs']['inp_steps']
    out_steps = test._kwargs['model_kwargs']['out_steps']

    # Gera uma lista de possíveis datas
    possible_dates = VWAP.index[inp_steps:]
    
    # Converte as datas para strings
    date_strings = possible_dates.sort_values(ascending=False).strftime('%d/%m/%Y').tolist()
    
    # obter a data mais atual
    try:
        data_atual = date_strings[-1]
    except:
        data_atual = None
    
    
    # obter diretorios
    try:
        arquivos = os.listdir(output_dir)
        
        arquivos = [arquivo.replace('rede_neural_', '').replace('.json', '') for arquivo in arquivos]
        
        # sort pela data (que é o nome do arquivo)
        arquivos = sorted(arquivos, key=lambda x: pd.to_datetime(x, format='%d_%m_%Y'))
    except Exception as e:
        print(e)
        arquivos = []
        
    
    return render_template("curto_prazo/curto_prazo_rede_neural.html", usuario=session["usuario"], menu=menu, arquivos=arquivos, dias=date_strings)

