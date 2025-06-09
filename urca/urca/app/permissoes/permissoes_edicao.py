from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/permissoes_edicao", methods=["GET"])
def permissoes_edicao():
    """Rota para mostrar a página de edição de permissões

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e redireciona caso não haja permissão
    3. Obtem os dados do formulário
    4. Define as variáveis, e gera as queries programaticamente, e sanitiza
    5. Obtém a contagem total de permissões
    6. Obtém os templates do banco de dados

    Nota: As queries em que contém 'grupos' faz referência às permissões

    Returns:
        Renderiza: A página de edição de permissões
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtem as permissoes do usuário e redireciona caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][2]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Obtém a permissão selecionada utilizando o ID salvo na sessão (no arquivo _permissoes_verifica.py)
    resposta = modulos.banco().sql(
        # "select id,str_nome,bol_status,fk_id_cliente, case when bol_status is true then \
        #                            'ATIVO' else 'INATIVO' end as str_status from grupos where id=$1;",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            id, \
            str_nome, \
            bol_status, \
            fk_id_cliente, \
            CASE WHEN bol_status IS true THEN 'ATIVO' ELSE 'INATIVO' END AS str_status \
        FROM grupos \
        WHERE \
            id = $1;",
        (session["edita_grupo"]),
    )
    if resposta:
        edita_grupo = resposta[0]

    # print(edita_grupo)

    # Obtém, da sessão, o filtro de TEMPLATES, se houver, para filtar pelos templates já selecionados
    if "lista_templates" in session.keys():
        if "not" in session["lista_templates"]:
            if session["usuario"]["admin"]:
                # Obtém dados do template/cliente
                resposta = modulos.banco().sql(
                    # "select id,str_nome from cliente where bol_status and not bol_template order by str_nome",
                    # Adicionando maiúsculas e aprimorando a sintaxe da query
                    "SELECT \
                        id, \
                        str_nome \
                    FROM cliente \
                    WHERE \
                        bol_status AND \
                        NOT bol_template \
                    ORDER BY str_nome",
                    (),
                )
                clientes = resposta
            else:
                clientes = []
        else:
            if session["usuario"]["admin"]:
                # Obtém dados do template/cliente
                resposta = modulos.banco().sql(
                    # "select id,str_nome from cliente where bol_status and bol_template order by str_nome",
                    # Adicionando maiúsculas e aprimorando a sintaxe da query
                    "SELECT \
                        id, \
                        str_nome \
                    FROM cliente \
                    WHERE \
                        bol_status AND \
                        bol_template \
                    ORDER BY str_nome",
                    (),
                )
                clientes = resposta
            else:
                clientes = []

    else:
        # Se não existir o 'lista_templates' na sessão
        if session["usuario"]["admin"]:
            # Obtém dados do template/cliente
            resposta = modulos.banco().sql(
                # "select id,str_nome from cliente where bol_status and not bol_template order by str_nome",
                # Adicionando maiúsculas e aprimorando a sintaxe da query
                "SELECT \
                    id, \
                    str_nome \
                FROM cliente \
                WHERE \
                    bol_status AND \
                    NOT bol_template \
                ORDER BY str_nome",
                (),
            )
            clientes = resposta
        else:
            clientes = []

    # Obtém todos os modulos
    modulos_grupos = modulos.banco().sql(
        # "select id,str_nome_modulo, bol_crud from modulos order by str_nome_modulo",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            id, \
            str_nome_modulo, \
            bol_crud \
        FROM modulos \
        ORDER BY str_nome_modulo",
        (),
    )

    modulos_dic = {}
    # Popula o dicionário de permissões com listas vazias
    for i in modulos_grupos:
        modulos_dic[i["id"]] = []

    # Obtém as permissões (SELECT, INSERT, UPDATE, DELETE) dos módulos, utilizando o ID do grupo/permissão
    g_a_m = modulos.banco().sql(
        # "select * from une_grupo_a_modulo where fk_id_grupo=$1",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            * \
        FROM une_grupo_a_modulo \
        WHERE \
            fk_id_grupo = $1",
        (session["edita_grupo"]),
    )
    # Popula o dicionário de permissões com as permissões de cada módulo (SELECT, INSERT, UPDATE, DELETE)
    for i in g_a_m:
        modulos_dic[i["fk_id_modulo"]] = [
            i["bol_s"],
            i["bol_i"],
            i["bol_u"],
            i["bol_d"],
        ]

    for j in modulos_grupos:
        if j["id"] not in modulos_dic.keys():
            # Define uma lista com tudo FALSE para o módulo, caso não exista
            modulos_dic[j["id"]] = [False, False, False, False]
        elif len(modulos_dic[j["id"]]) == 0:
            # Define uma lista com tudo FALSE para o módulo, caso não exista
            modulos_dic[j["id"]] = [False, False, False, False]
            
            # Atualiza as permissões no banco de dados, adicionando a permissão que não existe
            modulos.banco().sql(
                # "insert into une_grupo_a_modulo (fk_id_grupo,fk_id_modulo) values ($1,$2) returning id",
                # Adicionando maiúsculas e aprimorando a sintaxe da query
                "INSERT INTO une_grupo_a_modulo \
                    (fk_id_grupo, fk_id_modulo) \
                VALUES \
                    ($1, $2) \
                RETURNING id",
                (session["edita_grupo"], j["id"]),
            )

    return render_template(
        "permissoes/permissoes_edicao.html",
        usuario=session["usuario"],
        menu=menu,
        edita_grupo=edita_grupo,
        clientes=clientes,
        modulos=modulos_grupos,
        modulos_dic=modulos_dic,
        novo=0,
    )
