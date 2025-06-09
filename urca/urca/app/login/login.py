from flask import render_template, request, redirect, url_for, session
import re
import os
from datetime import datetime
from biblioteca.modulos import Banco, check
import json
import hashlib
from random import randint
from config.textos import MSG_ERROS


from app.login import b_login


@b_login.route("/login", methods=["GET"])
def login():
    """
    Inicializa o login adicionando o IP à sessão, e obtém a mensagem, se válida, da sessão.

    Renderiza o template "login/login.html".
    """

    if "msg" in session.keys():
        msg = session["msg"]
    else:
        msg = ""

    session.clear()

    if "X-Real-IP" not in request.headers.keys():
        session["endereco_ip"] = request.remote_addr
    else:
        session["endereco_ip"] = request.headers["X-Real-IP"]
        
    # check().check_modulos()  # Verifica os módulos iniciais, e adiciona se não existir
    
    
    return render_template("login/login.html", msg=msg)
