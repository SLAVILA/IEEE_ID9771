from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_altera_une_grupo_permissao", methods=["POST"])
def _permissoes_altera_une_grupo_permissao():
    """Rota para alterar todas as permissões (SELECT, INSERT, UPDATE, DELETE) da permissão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e retorna caso não haja permissão
    3. Obtem os dados do formulário
    4. Obtém o nome da permissão/grupo
    5. Obtém o nome do módulo
    6. Faz alterações dependendo do TIPO selecionado:
       - 't': Faz um loop para todas as permissões (SELECT, INSERT, UPDATE, DELETE), invertendo-as e salva no banco de dados
       - qualquer outro valor: Inverte a permissão selecionada, e salva no banco de dados
    7. Grava o log dependendo do TIPO
    8. Retorna com sucesso
    

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
    tipo = form.get("tipo")
    mod = form.get("mod")
    grupo = form.get("grupo")

    dic_tipos = {
        "s": "Visualizar",
        "i": "Incluir",
        "u": "Alterar",
        "d": "Excluir",
        "t": "Todas",
    }

    # Obtém o nome da permissão/grupo
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

    # Este if toma conta de ALTERAR O CRUD de um módulo, quando clicado no botão "É CRUD?"
    if not resposta:
        # Converte o True do python para JavaScript
        tipo = "True" if tipo == "true" else "False"

        # Altera o CRUD do módulo no banco de dados
        resposta = modulos.banco().sql(
            "UPDATE modulos \
            SET \
                bol_crud = $1 \
            WHERE \
                id = $2 \
            RETURNING bol_crud",
            (tipo, grupo),
        )
        retorno = {"status": "0" if resposta[0] else "1"}
        return json.dumps(retorno)

    # Obtém o nome do módulo
    resposta = modulos.banco().sql(
        # "select str_nome_modulo from modulos where id=$1",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            str_nome_modulo \
        FROM modulos \
        WHERE \
            id = $1",
        (mod),
    )
    modulo = resposta[0]["str_nome_modulo"]

    # Caso o tipo seja 't' (todos), inverte o CRUD dos módulos
    if tipo == "t":
        crud = ["bol_s", "bol_i", "bol_u", "bol_d"]

        # Faz um loop, para inverter cada permissão (SELECT, INSERT, UPDATE, DELETE) do módulo/permissão
        for permission in crud:
            modulos.banco().sql(
                # "update une_grupo_a_modulo set bol_s=not bol_s where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_s",
                # Adicionando maiúsculas e aprimorando a sintaxe da query
                "UPDATE une_grupo_a_modulo \
                SET \
                    %s = NOT %s \
                WHERE \
                    fk_id_modulo = $1 AND \
                    fk_id_grupo = $2 \
                RETURNING %s"
                % (permission),
                (mod, grupo),
            )

        # Removido código abaixo, aprimorando-o para um LOOP
        """
        resposta = modulos.banco().sql(
            # "update une_grupo_a_modulo set bol_s=not bol_s where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_s",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE une_grupo_a_modulo \
            SET \
                bol_s = not bol_s \
            WHERE \
                fk_id_modulo = $1 AND \
                fk_id_grupo = $2 \
            RETURNING bol_s",
            (mod, grupo),
        )
        resposta = modulos.banco().sql(
            # "update une_grupo_a_modulo set bol_i=not bol_i where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_i",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE une_grupo_a_modulo \
            SET \
                bol_i = not bol_i \
            WHERE \
                fk_id_modulo = $1 AND \
                fk_id_grupo = $2 \
            RETURNING bol_i",
            (mod, grupo),
        )
        resposta = modulos.banco().sql(
            "update une_grupo_a_modulo set bol_u=not bol_u where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_u",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE une_grupo_a_modulo \
            SET \
                bol_u = not bol_u \
            WHERE \
                fk_id_modulo = $1 AND \
                fk_id_grupo = $2 \
            RETURNING bol_u",
            (mod, grupo),
        )
        resposta = modulos.banco().sql(
            # "update une_grupo_a_modulo set bol_d=not bol_d where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_d",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE une_grupo_a_modulo \
            SET \
                bol_d = not bol_d \
            WHERE \
                fk_id_modulo = $1 AND \
                fk_id_grupo = $2 \
            RETURNING bol_d",
            (mod, grupo),
        )
        """
    else:
        # Se não for 't', apenas altera a permissão necessária
        resposta = modulos.banco().sql(
            # "update une_grupo_a_modulo set bol_%s=not bol_%s where fk_id_modulo=$1 and fk_id_grupo=$2 returning bol_%s"
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE une_grupo_a_modulo \
            SET \
                bol_%s = NOT bol_%s \
            WHERE \
                fk_id_modulo = $1 AND \
                fk_id_grupo = $2 \
            RETURNING \
                bol_%s"
            % (tipo, tipo, tipo),
            (mod, grupo),
        )

    # Grava o log dependendo do tipo
    if tipo == "t":
        modulos.log().log(
            codigo_log=5,
            ip=request.remote_addr,
            modulo_log="PERMISSOES",
            fk_id_usuario=session["usuario"]["id"],
            texto_log="Tipo de permissão alterado para True em todas as permissões de %s, modulo %s"
            % (nome, modulo),
        )
    else:
        modulos.log().log(
            codigo_log=5,
            ip=request.remote_addr,
            modulo_log="PERMISSOES",
            fk_id_usuario=session["usuario"]["id"],
            texto_log="Tipo de permissão %s alterado para %s na permissão %s, modulo %s"
            % (dic_tipos[tipo], resposta[0]["bol_%s" % (tipo)], nome, modulo),
        )
    
    # Retorna com sucesso
    retorno = {"status": "0"}
    return json.dumps(retorno)
