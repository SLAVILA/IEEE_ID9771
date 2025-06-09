from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/_config_modulos_novo", methods=["POST"])
def _config_modulos_novo():
    """
    Método POST para adicionar um novo módulo

    1. Obtém os dados do formulário
    2. Adiciona o novo módulo no banco de dados
    3. Retorna o resultado

    Returns:
        JSON: A resposta da adição do módulo
    """

    # Obtém os dados do formulário
    f = request.form
    nome = f.get("str_nome_modulo")
    descr = f.get("str_descr_modulo")

    # Adiciona o novo módulo no banco de dados
    resposta = modulos.banco().sql(
        #"INSERT INTO modulos (str_nome_modulo, str_descr_modulo) VALUES ($1, $2) returning id",
        # Adicionando maiúsculas e modificando a query para aprimorar a legibilidade
        "INSERT INTO modulos \
            (str_nome_modulo, str_descr_modulo) \
        VALUES ($1, $2) \
        RETURNING id",
        (nome, descr),
    )
    
    # Retorna o resultado, dependendo da resposta
    if resposta:
        session["msg"] = "Modulo inserido com sucesso!"
        session["msg_type"] = "success"
        retorno = {"status": "0"}
        return json.dumps(retorno)
    else:
        retorno = {"status": "1", "msg": "Erro ao inserir modulo..."}
        return json.dumps(retorno)
