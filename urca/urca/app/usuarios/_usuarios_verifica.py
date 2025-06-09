from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_verifica", methods=["POST"])
def _usuarios_verifica():
    """
    Atualiza o 'edita_usuario' da sessão atual, substituindo pelo usuário selecionado.
    
    1. Verifica se o usuário está logado.
    2. Obtem os dados do formulário.
    3. Define o ID do usuário que está sendo editado.
    4. Retorna um objeto JSON com o status "0".

    Retorna:
        Uma string JSON representando o status da verificação. Se o usuário estiver logado, o status é "0".
        Se o usuário não estiver logado, o status é "99".
    """

    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    # Obtem os dados do formulário
    form = request.form
    # id = form.get("id")
    # Alterando o nome 'id', que é utilizado pelo python, para 'form_id'
    form_id = form.get("id")

    # Define o ID do usuario que está sendo logado
    session["edita_usuario"] = form_id

    retorno = {"status": "0"}
    return json.dumps(retorno)
