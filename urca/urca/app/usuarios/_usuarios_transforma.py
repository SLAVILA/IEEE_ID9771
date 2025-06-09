from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_transforma", methods=["POST"])
def _usuarios_transforma():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    form = request.form

    # id = form.get("id")
    # Alterando o nome 'id', que é utilizado pelo Python, para 'form_id'
    form_id = form.get("id")

    session["super_usuario"] = session["usuario"]["id"]
    resposta = modulos.banco().sql(
        # "SELECT usuario.*, cliente.str_nome AS cliente, str_db_nome FROM usuario left JOIN cliente ON fk_id_cliente = cliente.id WHERE usuario.id=$1",
        # Adicionando maíusculas e aprimorando a visualização
        "SELECT \
            usuario.*, \
            cliente.str_nome AS cliente, \
            str_db_nome \
        FROM usuario \
        LEFT JOIN cliente \
            ON fk_id_cliente = cliente.id \
        WHERE \
            usuario.id = $1",
        (form_id),
    )
    if not resposta:
        # Se não encontrar o usuário pelo form_id
        retorno = {
            "status": "1",
            # "msg": MSG_ERROS["CPF_INSCRICAO_NAO_ENCONTRADO_NA_BASE"],
            # MSG_ERROS não está definido, alterando-o
            "msg": "Usuário selecionado para transformar não foi encontrado no banco de dados",
        }
        return json.dumps(retorno)

    else:
        # Se for um usuário válido
        if resposta[0]["bol_admin"]:
            id_cliente = 0
            nomecliente = "SUPERADMIN"
        else:
            id_cliente = resposta[0]["fk_id_cliente"]
            nomecliente = resposta[0]["cliente"]

        #modulos.config().verificar_config(id_cliente)

        # Carrega as informações do usuário requerido, substituindo os dados do usuário logado
        session["usuario"] = {
            "id": resposta[0]["id"],
            "nome": resposta[0]["str_nome"],
            "admin": resposta[0]["bol_admin"],
            "cadastro": resposta[0]["bol_cadastro"],
            "id_cliente": id_cliente,
            "nome_cliente": nomecliente,
            "id_tipo": resposta[0]["fk_id_tipo"],
            "email": resposta[0]["str_email"],
            "trocou_senha": resposta[0]["bol_trocou_senha"],
            "usuario_tag1": resposta[0]["usuario_tag1"],
            "usuario_tag2": resposta[0]["usuario_tag2"],
            "usuario_tag3": resposta[0]["usuario_tag3"],
            "usuario_tag4": resposta[0]["usuario_tag4"],
            "id_setor": resposta[0]["fk_id_setor"],
            "int_urna": resposta[0]["int_urna"],
            "db_nome": resposta[0]["str_db_nome"],
        }

        # log = modulos.log().log(
        # Removendo o retorno do log pois não é utilizado
        modulos.log().log(
            codigo_log=5,
            ip=session["endereco_ip"],
            modulo_log="LOGIN",
            fk_id_usuario=session["usuario"]["id"],
            fk_id_cliente=session["usuario"]["id_cliente"],
            texto_log="Usuário logou no admin.",
        )
    retorno = {"status": "0"}
    return json.dumps(retorno)
