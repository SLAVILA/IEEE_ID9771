import os
from flask import render_template, request, redirect, session
from datetime import datetime, timedelta
from biblioteca import modulos
import json
from ast import literal_eval
import psutil

from app.home import b_home
@b_home.route("/")
def home():
    if not 'usuario' in session.keys():
        return redirect("https://"+request.host+"/login")
    if not session['usuario']['trocou_senha']:
        return redirect("https://"+request.host+"/login_alteracao_senha")
    if 'sessao_selecionada' in session.keys():
        session.pop('/')
    if not 'id_cliente' in session['usuario'].keys():
        return redirect("https://"+request.host+"/login")
    if 'msg' in session.keys():
        msg = session['msg']
        session.pop('msg')
    else:
        msg = ''
    if 'msg_type' in session.keys():
        msg_type = session['msg_type']
        session.pop('msg_type')
    else:
        msg_type = ''
    

     # VERIFICA NGINX ESTÁ EM USO OU NÃO E DEFINE A SESSÃO endereco_ip
    # DESTA FORMA, CASO O FLAS ESTEJA SENDO EXECUTADO DIRETO, PARA DEBUG, A INFORMAÇÃO DE IP SEGUE SENDO ARMAZENADA CORRETAMENTE
    if not 'HTTP_CF_CONNECTING_IP' in request.environ.keys():
        if not 'HTTP_X_REAL_IP' in request.environ.keys():
            session['endereco_ip']=request.remote_addr
        else:
            session['endereco_ip']=request.environ["HTTP_X_REAL_IP"]
    else:
        session['endereco_ip']=request.environ["HTTP_CF_CONNECTING_IP"].split(',')[0]
        
    csv_path = os.path.join('app', 'curto_prazo', 'dados_preco', 'linear interpol', 'VWAP pentada', 'rolloff suavizado M+1 SE -> VWAP.csv')
    
    #ler a primeira e a ultima data do csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        first_date = lines[1].split(',')[1]
    
    # transformar em datas brasileiras
    start_date = datetime.strptime(first_date, '%Y-%m-%d')
    end_date = datetime.now() - timedelta(days=1)
    delta = end_date - start_date
    datas = [(start_date + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(delta.days + 1)]
    
    # sort pela data mais recente
    datas = sorted(datas, key=lambda x: datetime.strptime(x, '%d/%m/%Y'), reverse=True)
    

    if session['usuario']['admin']:
        menu=modulos.permissao().lista_permissao(session['usuario']['id'])
        
        
        disco = psutil.disk_usage('/')
        porcentagem_disco = disco.percent
        
        
        if porcentagem_disco >= 90:
            msg = "Alerta. Seu uso de disco excede o limite de 90%."
        

        return render_template('dashboards/dashboard.html',
            usuario=session['usuario'],menu=menu, msg=msg, datas=datas)
    else:

              

        config_cliente=modulos.config().cliente(session['usuario']['id_cliente'])
        
        session['openauth'] = False
        
        return render_template('dashboards/dashboard.html',usuario=session['usuario'],menu=menu,msg=msg,msg_type=msg_type,
                               config_cliente=config_cliente, datas=datas)





@b_home.route("/_mudar_banco", methods = ['POST'])
def _mudar_banco():
    # VALIDA SESSÃO
    if 'usuario' not in session.keys():
        retorno = {'status': '99', 'msg': 'Não está logado...'}
        return json.dumps(retorno)
    form=request.form

    banco_selecionado = form.get('database')
    
    dict_banco = literal_eval(banco_selecionado)
    
    url = request.host.split('.')[0]
    
    print(f"Banco selecionado: {dict_banco}")
    modulos.banco().mudar_banco(dict_banco['base'], host=dict_banco['host'], url=url)
    
    
    
    retorno = {'status': '0'}
    return json.dumps(retorno)