import atexit
import builtins
import inspect
import signal
from flask import Flask, copy_current_request_context, current_app, redirect, render_template, request, session
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit, join_room, leave_room
import logging
import os
import sys
import threading
from importlib import import_module
from config.config import CONFIG, CONFIG_TECH
from biblioteca import modulos
from biblioteca import generate_long_data
from biblioteca.task import Tasker
from app.curto_prazo import _markov as markov
from app.curto_prazo import _estocastico as estocastico
from app.curto_prazo import _analise_tecnica as analise
from app.longo_prazo import _longo_prazo as longo_prazo
import multiprocessing
from biblioteca.pre_processing.prices import PricesFillMissing
from biblioteca.pre_processing import Product_hip, Variable_hip
import numpy as np
import pandas as pd
import os
import warnings

# Store the original print function
original_print = builtins.print

def custom_print(*args, color=modulos.CoresConsole.GREEN, **kwargs):
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    modified_args = (f'[{datetime.now()}] {color}{caller_name}:',)
    modified_args += args
    modified_args += (modulos.CoresConsole.RESET,)
    original_print(*modified_args, **kwargs)

# Override the built-in print function
builtins.print = custom_print

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level_colors = {
            'ERROR': modulos.CoresConsole.RED,
            'WARNING': modulos.CoresConsole.YELLOW,
            'INFO': modulos.CoresConsole.BLUE,
            'PRINT': modulos.CoresConsole.GREEN
        }
        color = level_colors.get(record.levelname, '')
        record.msg = f"{color}{record.msg}{modulos.CoresConsole.RESET}"
        record.levelname = f"{color}{record.levelname}{modulos.CoresConsole.RESET}"
        return super().format(record)

def setup_logging():
    formatter = ColoredFormatter("[%(asctime)s] %(levelname)s: %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

class MyApp(Flask):
    socketio = None
    inicializado = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config['TEMPLATES_AUTO_RELOAD'] = True
        self.config['SECRET_KEY'] = 'dfgsdfg457y6345thgsdfgae46w476'
        self.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=360)
        self.config['SESSION_COOKIE_HTTPONLY'] = True
        self.config['REMEMBER_COOKIE_HTTPONLY'] = True 
        self.config['SESSION_COOKIE_SECURE'] = True
        self.config['REMEMBER_COOKIE_SECURE'] = True
        self.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
        self.config.update(SESSION_COOKIE_NAME='_urcaadm')
        self.register_blueprints()

        self.dados_historicos = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.estocastico_dash = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.markov_dash = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.estocastico_risco = multiprocessing.Manager().dict()  # Shared dictionary for historical data
        self.dados_obtidos = multiprocessing.Manager().Value('i', False)  # Shared boolean for data obtained status
        
        setup_logging()
        
        # Initialize SocketIO
        MyApp.socketio = SocketIO(self, cors_allowed_origins="*.urca.lk6.com.br")
        
        # Dictionary to maintain online clients
        self.online_clients = multiprocessing.Manager().dict()
        
        def exit(*args):
            Tasker().encerrar_processos("TODOS")
        
        # Registra a função para ser chamada na saída
        atexit.register(exit)
        
        Tasker().nova_task(funcao=self.atualizar_dados, usuario_id=1)
       
    
    def atualizar_banco(self):
        print(f"[PROCESSANDO] - Obtendo dados históricos...")
        dados = self.obter_dados_historicos(True)
        # Store the result in the shared dictionary
        for key, value in dados.items():
            self.dados_historicos[key] = value
        self.dados_obtidos.value = True
        print(f"[FINALIZADO] Dados históricos obtidos com sucesso!")
        # self.notify_clients("Dados históricos prontos para uso.")
        self.gerar_csv()
        dados['preco_h'].to_csv(os.path.join('app', 'curto_prazo', 'dados_analise_tecnica', 'Data', 'preco_bbce.csv'))

    def atualizar_dados(self):
        print(f"[PROCESSANDO] - Obtendo dados históricos...")
        dados = self.obter_dados_historicos(True)
        # Store the result in the shared dictionary
        for key, value in dados.items():
            self.dados_historicos[key] = value
        self.dados_obtidos.value = True
        print(f"[FINALIZADO] Dados históricos obtidos com sucesso!")
        # self.notify_clients("Dados históricos prontos para uso.")
        # self.gerar_csv()
        dados['preco_h'].to_csv(os.path.join('app', 'curto_prazo', 'dados_analise_tecnica', 'Data', 'preco_bbce.csv'))
    
    def iniciar_markov(self):
        print(f"[PROCESSANDO] - Calculando markov...")
        markov.calcular_markov()
    
    def iniciar_longo_prazo(self):
        print(f"[PROCESSANDO] - Calculando longo prazo...")
        generate_long_data.generate_all(os.path.join("app", "longo_prazo", "dados"))
        longo_prazo.longo_prazo()
    
    def iniciar_estocastico(self):
        print(f"[PROCESSANDO] - Calculando estocastico...")
        estocastico.estocastico(None, 15, 0.005, make_plot=False)
    
    
    def iniciar_analise(self):
        print(f"[PROCESSANDO] - Calculando analise backtest...")
        analise.execute_backtest()
    
    def gerar_csv(self):
        
    
        def get_precos(preco):
            price_dict = PricesFillMissing.get_all(preco)

            preco_clean = price_dict["raw"]
            preco_pent = price_dict["VWAP pentada"]
            preco_lin = price_dict["linear"]
            return preco_clean, preco_pent, preco_lin

        preco = self.obter_dados_historicos()['preco_h']
        preco_clean, preco_pent, preco_lin = get_precos(preco)

        varible_hip = Variable_hip.implemented()
        varible_hip["VWAP"] = lambda x: x

        directory = os.path.join('app', 'curto_prazo', 'dados_preco', 'linear interpol')
        for process_type, processado in (("raw", preco_clean), ("VWAP pentada", preco_pent), ("interpolacao linear", preco_lin)): 
            os.makedirs(os.path.join(directory, process_type), exist_ok=True)
            dados = {}
            count = 0
            for prod_hip in list(Product_hip.implemented().keys()):
                print(prod_hip)

                data = Product_hip.implemented()[prod_hip](processado.copy())
                idx = pd.Index(pd.date_range(data.data.min(), data.data.max()), name='data')

                data = data.set_index("data").reindex(idx, fill_value=None).reset_index()
                data["missing"] = None
                data["ok"] = None
                data.loc[data.VWAP.isna(), "missing"] = count-0.25
                data.loc[False==(data.VWAP.isna()), "ok"] = count
                count += 1

                for var_hip in list(varible_hip.keys()):
                    if var_hip in ("hip_12", "hip_15"):
                        if prod_hip.startswith("rolloff diferenca cumulativa"):
                            continue
                    name = f"{prod_hip} -> {var_hip}"
                    print(name)
                    df = varible_hip[var_hip](data.copy())
                    idx = pd.Index(pd.date_range(df.data.min(), df.data.max()), name='data')
                    df = df.set_index("data").reindex(idx, fill_value=None).reset_index()
                    dados[name] = df
                    print("\t", df.VWAP.isna().sum(), df[df.VWAP.isna()].data.max())
                    df.to_csv(os.path.join(directory, process_type, name+".csv"))
                    print(f"\t{name} salvo em: '{os.path.join(directory, process_type, name)}.csv'")
    
    def obter_estocastico_dash(self, usuario_id):
        return self.estocastico_dash.get(usuario_id, None)
    
    
    def obter_markov_dash(self, usuario_id):
        return self.markov_dash.get(usuario_id, None)
        
    def obter_dados_historicos(self, refresh=False):
        if refresh:
            preco, preco_h, preco_r, precobbce, subsistema, historico_hidrologia = modulos.Banco().get_precos()
            pld, pld_piso_teto, PLD = modulos.Banco().get_pld()
            self.dados_historicos["preco"] = preco
            self.dados_historicos["preco_h"] = preco_h
            self.dados_historicos["preco_r"] = preco_r
            self.dados_historicos["produtos"] = preco_h["produto"].unique()
            self.dados_historicos["pld"] = pld
            self.dados_historicos["PLD"] = PLD
            self.dados_historicos["pld_piso_teto"] = pld_piso_teto
            self.dados_historicos["precobbce"] = precobbce
            self.dados_historicos['subsistema'] = subsistema
            self.dados_historicos['historico_hidrologia'] = historico_hidrologia
            self.dados_historicos['ultimo_update'] = datetime.now()
        return self.dados_historicos

    def register_blueprints(self):
        diretorio_blueprints = 'app'
        exclusoes = ["__pycache__"]
        pacotes_blueprints = [d for d in os.listdir(diretorio_blueprints) if os.path.isdir(os.path.join(diretorio_blueprints, d)) and d not in exclusoes]
        
        for nome_pacote in pacotes_blueprints:
            modulo = import_module(f'{diretorio_blueprints}.{nome_pacote}')
            blueprint = getattr(modulo, f'b_{nome_pacote}', None)
            if blueprint and hasattr(blueprint, 'register'):
                self.register_blueprint(blueprint)
                #print(f'Blueprint registrado: {nome_pacote}')
            else:
                print(f"Blueprint {nome_pacote} não encontrado")
    
    # Define a function to notify clients
    def notify_clients(self, message, user_ids=None):
        try:
            if user_ids is None:
                MyApp.socketio.emit('notification', {'message': message})
            else:
                for user_id in user_ids:
                    sid = app.online_clients.get(user_id)['sid']
                    if sid:
                        MyApp.socketio.emit('notification', {'message': message}, room=sid)
            print(f"[NOTIFICAÇÃO] {message} - {user_ids}")
        except:
            pass
    
    # Define a function to notify clients
    def kick_clients(self, user_ids=None):
        for user_id in user_ids:
            sid = app.online_clients.get(user_id)['sid']
            if sid:
                MyApp.socketio.emit('kick', room=sid)

# Initialize Flask app
app = MyApp(__name__,
            static_url_path='',
            static_folder='../web/static',
            template_folder='../web/templates')

# Define WebSocket event handlers
@app.socketio.on('connect')
def handle_connect():
    app.online_clients[session['usuario']['id']] = {"sid": request.sid, "room": "room_" + str(session['usuario']['id'])}

@app.socketio.on('disconnect')
def handle_disconnect():
    user_id = session['usuario']['id']
    if user_id:
        app.online_clients.pop(user_id, None)
        print(f"[{session['usuario']['nome']}] Se desconectou do servidor.")

@app.socketio.on('join')
def handle_join(data):
    user_id = session['usuario']['id']
    room = app.online_clients[user_id]['room']
    if user_id and room:
        join_room(room)
        print(f"[{session['usuario']['nome']}] entrou no servidor.")
        app.notify_clients(f"[{session['usuario']['nome']}] conectado ao servidor.", user_ids=[user_id])

@app.socketio.on('leave')
def handle_leave(data):
    user_id = session['usuario']['id']
    room = app.online_clients[user_id]['room']
    if user_id and room:
        leave_room(room)
        print(f"[{session['usuario']['nome']}] saiu do servidor.")

# Define routes and error handlers
@app.route('/')
def inicio():
    return redirect("/")

@app.after_request
def add_header(r):
    
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['X-Content-Type-Options'] = 'nosniff'
    r.headers['Content-Security-Policy'] = "img-src 'self' blob: data:; default-src 'self' data: *.urca.lk6.com.br *.cloudflare.com fonts.googleapis.com fonts.gstatic.com www.google.com www.gstatic.com cdn.jsdelivr.net 'unsafe-inline' ws:"
    r.headers['X-Frame-Options'] = 'SAMEORIGIN'
    r.headers['Access-Control-Allow-Origin']= '*.urca.lk6.com.br'
    r.headers['Referrer-Policy']= 'same-origin'
    r.headers['Permissions-Policy']= 'geolocation=(self)'
    r.headers['Strict-Transport-Security'] = 'max-age=17280000; includeSubDomains'
    r.headers['X-XSS-Protection'] = '1; mode=block'
    return r

@app.errorhandler(404)
def handle_404_error(e):
    if not session.get('usuario', None):
        return redirect("login")
    
    # Obtém as permissões do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])

    # CARREGA AS CONFIGURACOES DE CLIENTE DO BANCO
    config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    return render_template(
        "erro404.html",
        usuario=session["usuario"],
        menu=menu,
        config_cliente=config_cliente,
    )

# Run the app with SocketIO
if __name__ == '__main__':
    MyApp.socketio.run(app, host='0.0.0.0', port=8000)
