from flask import render_template, request, redirect, url_for, session
import re
import os
from datetime import datetime
from biblioteca import modulos
import json
import hashlib
from random import randint
from config.textos import MSG_ERROS


from app.login import b_login


@b_login.route("/login_alteracao_senha", methods=["GET"])
def login_alteracao_senha():
    """
    Lida com a rota "/login_alteracao_senha".

    1. Se a chave "usuario" não estiver na sessão, redireciona o usuário para a página de login.
    2. Se a chave "sessao_selecionada" estiver na sessão, remove-a da sessão.
    3. Se a chave "id_cliente" não estiver no dicionário "usuario" na sessão, redireciona o usuário para a página de login.
    4. Renderiza o template "login/login_alteracao_senha.html".

    Parâmetros:
    - Nenhum
    """

    # if not 'usuario' in session.keys():
    # O uso de not in verifica, mais claramente, se a string não existe na lista/dicionário
    if "usuario" not in session.keys():
        return redirect("https://" + request.host + "/login")

    if "sessao_selecionada" in session.keys():
        session.pop("sessao_selecionada")

    # if not 'id_cliente' in session['usuario'].keys():
    # O uso de not in verifica, mais claramente, se a string não existe na lista/dicionário
    if "id_cliente" not in session["usuario"].keys():
        return redirect("https://" + request.host + "/login")

    return render_template("login/login_trocar_senha.html")
