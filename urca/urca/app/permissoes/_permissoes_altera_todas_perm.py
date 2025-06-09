from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_altera_todas_perm", methods=["POST"])
def _permissoes_altera_todas_perm():
    """Rota para alterar todas as permissões (SELECT, INSERT, UPDATE, DELETE) da permissão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e retorna caso não haja permissão
    3. Obtem os dados do formulário
    4. Obtém o nome da permissão
    5. Obtém os ID's dos módulos/permissões (para fazer o loop)
    6. Faz o loop e atualiza os status das permissões, invertendo seus valores

    Returns:
        Retorna: Status 0 (sucesso)
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtem as permissoes do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][2]:
        retorno = {"status": "1", "msg": "Usuário não pode mais alterar permissões..."}
        return json.dumps(retorno)

    # Obtem os dados do formulário
    form = request.form
    grupo = form.get("grupo")

    # Obtém o nome da permissão
    resposta = modulos.banco().sql(
        # "select str_nome from grupos where id=$1",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            str_nome \
        FROM grupos \
        WHERE \
            id = $1",
        (grupo),
    )
    nome = resposta[0]["str_nome"]

    # Obtém os IDs dos módulos
    modulo = modulos.banco().sql(
        # "select id from modulos",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            id \
        FROM modulos",
        (),
    )

    # Faz um loop nos módulos, invertendo seus valores (ex: se era False, vira True)
    for i in modulo:
        # Atualiza as permissões (SELECT, INSERT, UPDATE, DELETE) do grupo/permissão
        modulos.banco().sql(
            # "update une_grupo_a_modulo set bol_s=not bol_s,bol_i=not bol_i, bol_u=not bol_u, bol_d=not bol_d where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_s",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE une_grupo_a_modulo \
            SET \
                bol_s = not bol_s, \
                bol_i = not bol_i, \
                bol_u = not bol_u, \
                bol_d = not bol_d \
            WHERE \
                fk_id_modulo = $1 AND \
                fk_id_grupo = $2 \
            RETURNING bol_s",
            (i["id"], grupo),
        )
        
    # Grava o log
    modulos.log().log(
        codigo_log=5,
        ip=request.remote_addr,
        modulo_log="PERMISSOES",
        fk_id_usuario=session["usuario"]["id"],
        texto_log="Alterado status de todos os módulos da permissão %s." % (nome),
    )
    
    # Retorna com sucesso
    retorno = {"status": "0"}
    return json.dumps(retorno)
