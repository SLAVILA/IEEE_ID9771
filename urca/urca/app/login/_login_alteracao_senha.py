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


@b_login.route("/_login_alteracao_senha", methods=["POST"])
def _login_alteracao_senha():
    """
    Lida com a alteração de senhas de login.
    Verifica se os campos foram preenchidos e se as senhas conferem.
    
    1. Verifica se o método da requisição é POST
    2. Recupera os dados do formulário
    3. Verifica se as senhas são válidas
    4. Encripta as senhas
    5. Altera a senha no banco de dados
    
    Parâmetros:
    - Nenhum
    
    Retorna:
    - Uma string JSON contendo uma mensagem e um código de status.
    """
    
    # if not request.method == "POST":
    # Para comparações de valores, onde se quer verificar se dois objetos tem valores diferentes, 
    # é mais comum usar os operadores de igualdade ou desigualdade (== e !=).
    if request.method != "POST":
        return redirect("https://" + request.host + "/login")
    else:
        form = request.form
        senha_atual = form.get("senhaatual").strip()
        nova_senha = form.get("senha1").strip()  # era senha1, alterada para melhorar legibilidade
        confirmar_senha = form.get("senha2").strip()  # era senha2, alterada para melhorar legibilidade

        if len(senha_atual) == 0:
            retorno = {"msg": "Informe a senha atual...", "status": "1"}
            return json.dumps(retorno)

        if len(nova_senha) == 0:
            retorno = {"msg": "Informe uma senha...", "status": "1"}
            return json.dumps(retorno)

        if len(nova_senha) < 6:
            retorno = {
                "msg": "Senha deve ter pelo menos 6 caracteres...",
                "status": "1",
            }
            return json.dumps(retorno)

        # if not senha1 == senha2:
        # Para comparações de valores, onde se quer verificar se dois objetos tem valores diferentes, 
        # é mais comum usar os operadores de igualdade ou desigualdade (== e !=).
        if nova_senha != confirmar_senha:
            retorno = {"msg": "Senha e confirmação devem ser iguais...", "status": "1"}
            return json.dumps(retorno)

        senha_atual = hashlib.sha512(senha_atual.encode()).hexdigest()
        resposta = modulos.banco().sql(
            # "select str_senha from  usuario where id=$1",
            # Modificando as queries, adicionando maiúsculas para facilitar o entendimento
            "SELECT \
                str_senha \
            FROM usuario \
            WHERE \
                id=$1",
            (session["usuario"]["id"]),
        )
        if resposta:
            senha = resposta[0]["str_senha"]
            # if not senhaatual == senha:
            # Para comparações de valores, onde se quer verificar se dois objetos tem valores diferentes, 
            # é mais comum usar os operadores de igualdade ou desigualdade (== e !=).
            if senha_atual != senha:
                retorno = {"msg": "Senha atual não confere...", "status": "1"}
                return json.dumps(retorno)

        senha_enc = hashlib.sha512(nova_senha.encode()).hexdigest()
        resposta = modulos.banco().sql(
            # "update usuario set str_senha=$1,bol_trocou_senha='t' where id=$2 returning id",
            # Modificando as queries, adicionando maiúsculas para facilitar o entendimento
            "UPDATE usuario \
            SET \
                str_senha=$1, \
                bol_trocou_senha='t' \
            WHERE id=$2 RETURNING id",
            (senha_enc, session["usuario"]["id"]),
        )
        session["usuario"]["trocou_senha"] = "t"
        session.modified = True
        # print(resposta)

        if not resposta:
            retorno = {"msg": "Erro ao alterar a senha...", "status": "1"}
            return json.dumps(retorno)
        else:
            retorno = {"status": "0"}
            return json.dumps(retorno)
