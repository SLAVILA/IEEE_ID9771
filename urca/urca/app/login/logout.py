from flask import render_template, request, redirect, url_for, session
import re
import os
from datetime import datetime
from biblioteca import modulos
import json
import hashlib
from random import randint
from config.textos import MSG_ERROS


from app.login import b_login


@b_login.route("/logout")
def logout():
    """
    Cuida do processo de logout do usuário.

    Parâmetros:
    - Nenhum

    Retorna:
        redirect: Redireciona o usuário para a página apropriada com base no seu status de login (super usuário ou usuário).
    """

    if "usuario" in session.keys():
        resposta = modulos.banco().sql(
            # "SELECT str_nome from usuario where id=$1",
            # Modificando as queries, adicionando maiúsculas para facilitar o entendimento
            "SELECT \
                str_nome \
            FROM usuario \
            WHERE \
                id = $1",
            (session["usuario"]["id"]),
        )
        try:
            # log = modulos.log().log(
            # Removendo o retorno do log, que não é utilizado
            modulos.log().log(
                codigo_log=5,
                ip=session["endereco_ip"],
                modulo_log="LOGOUT",
                fk_id_usuario=session["usuario"]["id"],
                fk_id_cliente=session["usuario"]["id_cliente"],
                texto_log="Usuário %s saiu do sistema." % (resposta[0]["str_nome"]),
            )
        except:
            pass

    if "super_usuario" in session.keys():
        super = session["super_usuario"]
        session.pop("super_usuario")

        resposta = modulos.banco().sql(
            # "SELECT usuario.*, cliente.str_nome AS cliente FROM usuario left JOIN cliente ON fk_id_cliente = cliente.id WHERE usuario.id=$1",
            # Modificando as queries, adicionando maiúsculas para facilitar o entendimento
            "SELECT \
                usuario.*, \
                cliente.str_nome AS cliente \
            FROM usuario \
            LEFT JOIN cliente \
                ON fk_id_cliente = cliente.id \
            WHERE \
                usuario.id = $1",
            (super),
        )
        if not resposta:
            retorno = {
                "status": "1",
                "msg": MSG_ERROS["CPF_INSCRICAO_NAO_ENCONTRADO_NA_BASE"],
            }
            session.clear()
            return redirect("https://" + request.host + "/login")
            return json.dumps(retorno)
        else:
            if resposta[0]["bol_admin"]:
                id_cliente = 0
                nomecliente = "SUPERADMIN"
            else:
                id_cliente = resposta[0]["fk_id_cliente"]
                nomecliente = resposta[0]["cliente"]

            session["usuario"] = {
                "id": resposta[0]["id"],
                "nome": resposta[0]["str_nome"],
                "admin": resposta[0]["bol_admin"],
                "id_cliente": id_cliente,
                "nome_cliente": nomecliente,
                "id_tipo": resposta[0]["fk_id_tipo"],
                "email": resposta[0]["str_email"],
                "usuario_tag1": resposta[0]["usuario_tag1"],
                "usuario_tag2": resposta[0]["usuario_tag2"],
                "usuario_tag3": resposta[0]["usuario_tag3"],
                "usuario_tag4": resposta[0]["usuario_tag4"],
                "trocou_senha": True,
                "int_urna": resposta[0]["int_urna"],
            }

            if "sessao_selecionada" in session.keys():
                # Remove a sessao_selecionada da sessão, evitando que usuários entrem onde não devem
                session.pop("sessao_selecionada")

        return redirect("https://" + request.host + "/usuarios_lista")
    else:
        session.clear()
        return redirect("https://" + request.host + "/login")
