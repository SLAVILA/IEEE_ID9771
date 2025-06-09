from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


@b_config_modulos.route("/_config_modulos_exclui", methods=["POST"])
def _config_modulos_exclui():
    """Exclui o módulo do banco de dados

    1. Verifica se o usuário está logado
    2. Obtém as permissões do usuário, e redirige caso não haja permissão
    3. Verifica se o 'exclui_modulo' está na sessão
    4. Obtém os dados do formulário, e os tokens
    5. Se os tokens forem iguais, exclui o módulo
    6. Retorna o resultado

    Returns:
        JSON: O resultado da remoção do módulo
    """
    if request.method == "POST":
        # Verifica se o usuário está logado
        # if not 'usuario' in session.keys():
        # O uso de not in aumenta a clareza ao ler a sintaxe
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)

        # Obtém as permissões do usuário, e redirige caso não haja permissão
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        if not menu[2][3]:
            retorno = {"status": "1", "msg": "Usuário não pode mais excluir modulo..."}
            return json.dumps(retorno)

        # Verifica se o 'exclui_modulo' está na sessão, e redireciona caso não esteja
        #if not "exclui_modulo" in session.keys():
        # O uso de not in aumenta a clareza ao ler a sintaxe
        if "exclui_modulo" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)
        else:
            # Obtém os dados do formulário
            form = request.form
            # Obtém os tokens salvos anteriormente, na função _config_modulos_verifica_exc.py
            token = form.get("token")
            token_sys = session["exclui_modulo"]["token"]

            # Se forem iguais, exclui o módulo
            if int(token) == token_sys:
                id_exc = session["exclui_modulo"]["id"]

                # Exclui o módulo do banco de dados
                resposta = modulos.banco().sql(
                    # "delete from modulos where id=$1 returning id",
                    # Adicionando maiúsculas e modificando a query para aprimorar a legibilidade
                    "DELETE FROM modulos \
                    WHERE \
                        id = $1 \
                    RETURNING id",
                    (id_exc),
                )
                # Retorna o resultado
                if resposta:
                    session["msg"] = "Modulo excluído com sucesso"
                    session["msg_type"] = "success"
                    retorno = {"status": "0"}
                    return json.dumps(retorno)
            else:
                # Token inválido
                retorno = {"status": "1", "msg": "Token incorreto..."}
                return json.dumps(retorno)
