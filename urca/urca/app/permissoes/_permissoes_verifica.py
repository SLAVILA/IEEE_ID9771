from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_verifica", methods=["POST"])
def _permissoes_verifica():
    """Rota para adicionar o ID da permissão selecionada à sessão

    1. Obtém os dados do formulário
    2. Adiciona o ID à sessão
    3. Retorna

    Returns:
        Retorna: Sucesso (status 0)
    """
    
    # Obtém os dados do formulário
    form = request.form
    form_id = form.get("id")
    
    # Adiciona o ID à sessão
    session["edita_grupo"] = form_id
    
    # Retorna
    retorno = {"status": "0"}
    return json.dumps(retorno)
