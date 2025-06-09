from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/config_modulos_novo", methods=["GET"])
def config_modulos_novo():
    """
    Renderiza a página de adição de módulos

    1. Verifica se o usuário está logado
    2. Obtém as permissões do usuário, e redirige caso não haja permissão
    3. Gera um novo módulo vazio
    4. Renderiza a pagina de edição do módulo com os dados (vazios) do novo módulo

    Returns:
        Rennderiza: A página da adição do módulo
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtém a permissão do usuário, e redirige caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[2][2]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Gera um novo módulo vazio
    resposta = {"str_nome_modulo": "", "str_descr_modulo": ""}

    # Renderiza a pagina de adição do módulo com os dados (vazios) do novo módulo
    return render_template(
        "config_modulos/config_modulos_edicao.html",
        modulo=resposta,
        usuario=session["usuario"],
        menu=menu,
        novo=1,
    )
