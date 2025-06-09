from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/_config_modulos_edicao", methods=["POST"])
def _config_modulos_edicao():
    """
    Método POST para editar um módulo

    1. Obtém os dados do formulário
    2. Adiciona o novo módulo no banco de dados
    3. Retorna o resultado

    Returns:
        JSON: A resposta da edição do módulo
    """

    edita_config_modulo = 0

    # Obtém, da sessão, o ID do módulo editado, e a remove da sessão
    if "edita_config_modulo" in session.keys():
        edita_config_modulo = session["edita_config_modulo"]
        session.pop("edita_config_modulo")

    # Obtém os dados do formulário
    f = request.form
    nome = f.get("str_nome_modulo")
    descr = f.get("str_descr_modulo")

    # Atualiza os dados do módulo no banco de dados utilizando o ID obtido da sessão
    resposta = modulos.banco().sql(
        #"UPDATE modulos SET str_nome_modulo=$1, str_descr_modulo=$2 WHERE id=$3 returning id",
        # Adicionando maiúsculas e modificando a query para aprimorar a legibilidade
        "UPDATE modulos \
        SET \
            str_nome_modulo = $1, \
            str_descr_modulo = $2 \
        WHERE \
            id=$3 \
        RETURNING id",
        (nome, descr, edita_config_modulo),
    )
    
   # Retorna o resultado, dependendo da resposta
    if resposta:
        session["msg"] = "Modulo alterado com sucesso!"
        session["msg_type"] = "success"
        retorno = {"status": "0"}
        return json.dumps(retorno)
    else:
        retorno = {"status": "1", "msg": "Erro ao alterar..."}
        return json.dumps(retorno)
