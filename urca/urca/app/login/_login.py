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


@b_login.route("/_login", methods=["POST"])
def _login():
    """
    Autentica as credenciais de login do usuário.

    1. Verifica se o método da requisição é POST
    2. Recupera os dados do formulário
    3. Verifica se o campo de login/email foi preenchido
    4. Verifica se o campo de senha foi preenchido
    5. Verifica se o e-mail está no banco de dados
    6. Verifica se a senha estiver correta
    7. Loga o usuário

    Parâmetros:
    - Nenhum

    Lógica:
        - Se o método da requisição não for "POST", redireciona o usuário para a página de login.
        - Se o campo de login estiver vazio, retorna uma resposta JSON com uma mensagem de erro.
        - Se o e-mail não for encontrado no banco de dados, retorna uma resposta JSON com uma mensagem de erro.
        - Se a senha estiver incorreta, retorna uma resposta JSON com uma mensagem de erro.
        - Se o login for bem-sucedido, atualiza a sessão do usuário e retorna uma resposta JSON com um status de sucesso.
    """

    # if not request.method == "POST":
    # O uso de != deixa mais clara a leitura, e faz parte das diretrizes do PEP 8 do Python
    if request.method != "POST":
        return redirect("https://" + request.host + "/login")
    else:
        form = request.form
        login = form.get("login").strip()
        if len(login) == 0:
            retorno = {
                "msg": "O campo login/email não pode ser vazio...",
                "status": "1",
            }
            return json.dumps(retorno)
        else:

            resposta = modulos.banco().sql(
                "SELECT \
                    id, \
                    bol_admin, \
                    bol_cadastro \
                FROM usuario \
                WHERE \
                    str_email ILIKE $1 \
                    AND bol_status",
                (login),
            )
            print(resposta)

            if not resposta:
                retorno = {"msg": "Email não encontrado na base", "status": "1"}
                
                return json.dumps(retorno)
            else:
                if "usuario" not in session.keys():  # Adiciona o usuario na sessão
                    session["usuario"] = {
                        "id": resposta[0]["id"],
                        "admin": resposta[0]["bol_admin"],
                        "cadastro": resposta[0]["bol_cadastro"],
                    }
                else:  # Modifica o usuário da sessão
                    session["usuario"]["id"] = resposta[0]["id"]
                    session["usuario"]["admin"] = resposta[0]["bol_admin"]
                    session["usuario"]["cadastro"] = resposta[0]["bol_cadastro"]
                session.modified = True

                senha = form.get("senha")
                resposta = modulos.banco().sql(
                    # Modificando as queries, adicionando maiúsculas para facilitar o entendimento
                    "SELECT \
                        usuario.*, \
                        cliente.str_nome AS cliente, \
                        cliente.id AS id_cliente, \
                        cliente.str_logo AS logo_cliente, \
                        str_db_nome \
                    FROM usuario \
                    LEFT JOIN cliente \
                        ON fk_id_cliente = cliente.id \
                    WHERE usuario.id=$1",
                    (session["usuario"]["id"]),
                )
                if not resposta:
                    retorno = {"status": "1", "msg": "Usuário não encontrado..."}
                    return json.dumps(retorno)
                else:
                    senha_banco = resposta[0]["str_senha"]

                    if senha_banco == hashlib.sha512(senha.encode()).hexdigest():
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
                            "cadastro": resposta[0]["bol_cadastro"],
                            "trocou_senha": resposta[0]["bol_trocou_senha"],
                            "id_cliente": id_cliente,
                            "nome_cliente": nomecliente,
                            "id_tipo": resposta[0]["fk_id_tipo"],
                            "email": resposta[0]["str_email"],
                            "usuario_tag1": resposta[0]["usuario_tag1"],
                            "usuario_tag2": resposta[0]["usuario_tag2"],
                            "usuario_tag3": resposta[0]["usuario_tag3"],
                            "usuario_tag4": resposta[0]["usuario_tag4"],
                            "db_nome": resposta[0]["str_db_nome"],
                        }
                        session.modified = True

                        retorno = {"status": "0"}
                       
                    else:
                        retorno = {"status": "1", "msg": "Senha Inválida..."}

                       

                    return json.dumps(retorno)

                