from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/_config_modulos_verifica", methods=["POST"])
def _config_modulos_verifica():
    """Obtém o ID do módulo selecionado

    1. Verifica se o usuário está logado
    2. Obtém os dados do formulário
    3. Adiciona na sessão o ID do módulo que será editado
    4. Retorna 0, sucesso

    Returns:
        JSON: Status 0 (sucesso)
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    # Obtém os dados do formulário
    form = request.form
    # id = form.get("id")
    # Alterado 'id' para 'form_id'
    form_id = form.get("id")

    # Adiciona na sessão o ID do módulo que será editado
    session["edita_config_modulo"] = form_id
    print(session["edita_config_modulo"])

    # Retorna 0, sucesso
    retorno = {"status": "0"}
    return json.dumps(retorno)
