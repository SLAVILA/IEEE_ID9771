from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/modulos_lista", methods=["GET"])
def config_modulos_lista():
    """
    Método para renderizar a pagina dos módulos

    1. Verifica se o usuário está logado
    2. Obtém a mensagem na sessão, e as remove da sessão
    3. Obtém a permissão do usuário
    4. Renderiza a página dos módulos

    Returns:
        Renderiza: A página dos módulos
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")
    
    # Obtém a mensagem na sessão, e as remove da sessão
    if "msg" in session.keys():
        msg = session["msg"]
        msg_type = session["msg_type"]
        session.pop("msg")
        session.pop("msg_type")
    else:
        msg = ""
        msg_type = ""
        
    # Obtém a permissão do usuário
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    # Renderiza a página dos módulos
    return render_template(
        "config_modulos/config_modulos_lista.html",
        usuario=session["usuario"],
        menu=menu,
        msg=msg,
        msg_type=msg_type,
    )
