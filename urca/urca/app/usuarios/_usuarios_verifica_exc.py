from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_verifica_exc", methods=["POST"])
def _usuarios_verifica_exc():
    """
    Atualiza o 'exclui_usuario' da sessão atual com um token aleatório, para que o usuário seja excluído com segurança.

    1. Verifica se o usuário está logado.
    2. Obtem os dados do formulário.
    3. Gera um token aleatório e salva na sessão, para não haver modificações no ID do usuário, junto com o ID do usuário que foi selecionado para excluir
    4. Marca a sessão como modificada
    5. Retorna um objeto JSON com o status "0".

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

    # Gera um token aleatório e salva na sessão, junto com o ID do usuário que foi selecionado para entrar
    # O token aleatório para mascarar o ID real. Este token será usado ao excluir o usuário, previnindo ataques.
    token = randint(10000000000, 99999999999)
    session["exclui_usuario"] = {"id": form_id, "token": token}

    # Marca a sessão como modificada
    session.modified = True

    # Retorna um objeto JSON com o status "0"
    retorno = {"status": "0", "token": token}
    return json.dumps(retorno)
