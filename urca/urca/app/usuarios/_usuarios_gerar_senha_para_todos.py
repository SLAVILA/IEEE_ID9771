from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_gerar_senha_para_todos", methods=["POST"])
def _usuarios_gerar_senha_para_todos():
    """
    Gera uma nova senha para todos os usuários, atualiza no banco de dados e envia o e-mail para cada um.
    
    1. Obtém todos os usuários
    2. Gera nova senha para cada usuário, atualiza no banco de dados e envia o e-mail
    3. Retorna uma string JSON contendo o status, mensagem e tipo de mensagem.

    Retorna:
        str: Uma string JSON contendo o status, mensagem e tipo de mensagem.
    """
    
    print("_todos_gerar_senha_para_todos")

    # Obtém todos os usuários
    rows_usuarios = modulos.banco().sql(
        # "select * from usuario where fk_id_cliente = $1 order by str_nome",
        # Adicionando maíusculas e aprimorando a visualização
        "SELECT \
            * \
        FROM usuario \
        WHERE \
            fk_id_cliente = $1 \
        ORDER BY str_nome",
        (session["usuario"]["id_cliente"]),
    )

    tmp_sucessos = 0
    tmp_falhas = 0

    session["tmp_total"] = 0
    session["tmp_enviados"] = 0
    session["tmp_total"] = len(rows_usuarios)

    # Gera nova senha para cada usuário, atualiza no banco de dados e envia o e-mail
    for usuarios in rows_usuarios:
        senha_gerada = modulos.senha().gerar()
        senha_gerada_encriptada = modulos.criptografia().encriptar(senha_gerada)

        tmp_update_senha = modulos.banco().sql(
            #"update usuario set str_senha = $1 where id = %s returning id ",
            # Adicionando maíusculas e aprimorando a visualização
            "UPDATE usuario \
            SET \
                str_senha = $1 \
            WHERE \
                id = %s \
            RETURNING id ",
            (senha_gerada_encriptada, usuarios["id"]),
        )

        if tmp_update_senha[0]["id"] > 0:
            tmp_sucessos += 1
            session["tmp_enviados"] += 1
        else:
            tmp_falhas += 1

    if len(rows_usuarios) > 0:
        session["msg"] = "Todas senhas geradas. Sucessos: " + str(tmp_sucessos)
        session["msg_type"] = "success"
    else:
        session["msg"] = "Falhou ao gerar todas senhas! Falhas: " + str(tmp_falhas)
        session["msg_type"] = "danger"

    retorno = {"status": "0", "msg": session["msg"], "msg_type": session["msg_type"]}

    return json.dumps(retorno)
