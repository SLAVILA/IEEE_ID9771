from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_altera_status", methods=["POST"])
def _permissoes_altera_status():
    """Rota para alterar o nome da permissão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e retorna caso não haja permissão
    3. Obtem os dados do formulário
    4. Altera o status da permissão
    5. Retorna com sucesso

    Returns:
        Retorna: Status 0 (sucesso)
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    # Obtém as permissoes do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][2]:
        retorno = {"status": "1", "msg": "Usuário não pode mais alterar permissões..."}
        return json.dumps(retorno)

    # Obtem os dados do formulário
    form = request.form
    grupo = form.get("grupo")

    # Altera o status da permissão
    resposta = modulos.banco().sql(
        # "update grupos set bol_status=not bol_status where id=$1 returning str_nome,bol_status",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "UPDATE grupos \
        SET \
            bol_status = not bol_status \
        WHERE \
            id = $1 \
        RETURNING \
            str_nome, \
            bol_status",
        (grupo),
    )
    # Grava o log
    modulos.log().log(
        codigo_log=5,
        ip=request.remote_addr,
        modulo_log="PERMISSOES",
        fk_id_usuario=session["usuario"]["id"],
        texto_log="Status da permissão %s alterado para %s"
        % (resposta[0]["str_nome"], resposta[0]["bol_status"]),
    )

    # Retorna com sucesso
    retorno = {"status": "0"}
    return json.dumps(retorno)
