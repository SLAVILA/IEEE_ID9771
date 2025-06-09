from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/config_modulos_edicao", methods=["GET"])
def config_modulos_edicao():
    """
    Renderiza a página de edição de módulos

    1. Verifica se o usuário está logado
    2. Obtém as permissões do usuário, e redirige caso não haja permissão
    3. Obtém, da sessão, o ID do módulo editado
    4. Obtém os dados do módulo do banco de dados usando o ID
    5. Renderiza a pagina de edição do módulo com os dados do módulo

    Returns:
        Rennderiza: A página da edição do módulo
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtém as permissões do usuário, e redirige caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[2][2]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    edita_config_modulo = 0

    # Obtém, da sessão, o ID do módulo editado
    if "edita_config_modulo" in session.keys():
        edita_config_modulo = session["edita_config_modulo"]
        # session.pop('edita_config_modulo')

    # Obtém os dados do módulo do banco de dados usando o ID salvo na sessão
    resposta = modulos.banco().sql(
        # "SELECT * FROM modulos WHERE id = $1",
        # Adicionando maiúsculas e modificando a query para aprimorar a legibilidade
        "SELECT \
            * \
        FROM modulos \
        WHERE \
            id = $1",
        (edita_config_modulo),
    )

    resposta = resposta[0]

    # Renderiza a página de edição do módulo
    return render_template(
        "config_modulos/config_modulos_edicao.html",
        modulo=resposta,
        usuario=session["usuario"],
        menu=menu,
        edita_config_modulo=edita_config_modulo,
        novo=0,
    )
