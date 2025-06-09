from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/_config_modulos_verifica_exc", methods=["POST"])
def _config_modulos_verifica_exc():
    """
    Atualiza o 'exclui_modulo' da sessão atual com um token aleatório, para que o usuário seja excluído com segurança.

    1. Verifica se o usuário está logado.
    2. Obtem os dados do formulário.
    3. Gera um token aleatório e salva na sessão, para não haver modificações no ID do usuário, junto com o ID do usuário que foi selecionado para excluir
    4. Marca a sessão como modificada
    5. Retorna um objeto JSON com o status "0".

    Retorna:
        JSON: Status 0 (sucesso), e o token gerado
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

    # Gera um token aleatório para mascarar o ID real. Este token será usado ao excluir o módulo, previnindo ataques.
    token = randint(10000000000, 99999999999)

    # Adiciona o token na sessão, e a marca como modificada
    session["exclui_modulo"] = {"id": form_id, "token": token}
    session.modified = True

    # Retorna o token
    retorno = {"status": "0", "token": token}
    return json.dumps(retorno)
