from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_lista", methods=["POST"])
def _permissoes_lista():
    """Rota para mostrar a lista de permissões

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e redireciona caso não haja permissão
    3. Obtem os dados do formulário
    4. Define as variáveis, e gera as queries programaticamente, e sanitiza
    5. Obtém a contagem total de permissões
    6. Obtém os templates do banco de dados

    Nota: As queries em que contém 'grupos' faz referência às permissões

    Returns:
        Renderiza: A lista de permissões
    """

    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtem as permissoes do usuário, e redireciona caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Obtem os dados do formulário
    f = request.form

    # Define as variáveis, e gera as queries programaticamente
    page = 1
    limit = 10
    contagem_total = 0
    total_pages = 0
    sort_field = ""
    sort_ord = "asc"
    search_query = ""
    search_query_sql = ""
    order_query = ""
    status_query_sql = ""

    if "lista_templates" in session.keys():
        template_query_sql = session["lista_templates"]
    else:
        template_query_sql = " AND not c.bol_template "

    if "pagination[page]" in f.keys():
        page = int(f.get("pagination[page]"))

    if "pagination[perpage]" in f.keys():
        limit = int(f.get("pagination[perpage]"))

    if "sort[field]" in f.keys():
        sort_field = f.get("sort[field]")

    if "sort[sort]" in f.keys():
        sort_ord = f.get("sort[sort]")
        order_query = "ORDER BY " + sort_field + " " + sort_ord

    if "query[generalSearch]" in f.keys():
        search_query = f.get("query[generalSearch]")
        if len(search_query) > 2:
            search_query_sql = " AND g.str_nome ILIKE '%" + search_query + "%' "

    if "query[Status]" in f.keys():
        status_query = f.get("query[Status]")
        if status_query == "1":
            status_query_sql = " AND g.bol_status "
        elif status_query == "2":
            status_query_sql = " AND not g.bol_status  "

    if "query[Templates]" in f.keys():
        template_query = f.get("query[Templates]")
        if template_query == "0":
            template_query_sql = " AND not c.bol_template "
        elif template_query == "1":
            template_query_sql = " AND c.bol_template  "

        session["lista_templates"] = template_query_sql
        session.modified = True

    # Sanitiza os queries
    (
        (
            template_query_sql,  # Se tiver valor, filtra pelo template
            search_query_sql,  # Se tiver valor, filtra pelo termo de busca
            status_query_sql,  # Se tiver valor, filtra pelo status
            order_query,  # Se tiver valor, filtra pela ordem
            limit,  # Quantidade de registros por página
            page,  # Pagina sendo exibida
            limit,  # Quantidade de registros por página (cálculo do offset)
        ),
        query_perigoso,  # IMPORTANTE: QUERY PERIGOSO É GERADO QUANDO A QUERY POSSUI EXPRESSÕES PERIGOSAS!
    ) = modulos.query().sanitizar(
        template_query_sql,
        search_query_sql,
        status_query_sql,
        order_query,
        limit,
        page,
        limit,
    )

    # Obtem a contagem total de permissões
    if session["usuario"]["admin"]:
        contagem = modulos.banco().sql(
            # "select count(g.id) from grupos g left join cliente c on c.id=g.fk_id_cliente where g.id > 0 %s %s %s"
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "SELECT \
                COUNT(g.id) \
            FROM grupos g \
            LEFT JOIN cliente c \
                ON c.id = g.fk_id_cliente \
            WHERE \
                g.id > 0 \
            %s %s %s"
            % (search_query_sql, template_query_sql, status_query_sql),
            (),
        )
    else:
        contagem = modulos.banco().sql(
            # "select count(g.id) from grupos g left join cliente c on c.id=g.fk_id_cliente where g.fk_id_cliente=$1 %s %s %s"
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "SELECT \
                COUNT(g.id) \
            FROM grupos g \
            LEFT JOIN cliente c \
                ON c.id = g.fk_id_cliente \
            WHERE \
                g.fk_id_cliente = $1 \
            %s %s %s"
            % (search_query_sql, template_query_sql, status_query_sql),
            (session["usuario"]["id_cliente"]),
        )

    contagem_total = contagem[0]["count"]
    total_pages = math.floor(contagem_total / limit)

    # Obtem as permissões do banco de dados
    if session["usuario"]["admin"]:
        resposta = modulos.banco().sql(
            # "select g.id,g.str_nome, c.str_nome as nome_cliente,g.bol_status from grupos g left join cliente c on c.id=g.fk_id_cliente where g.id > 0 %s %s %s %s LIMIT %s \
            #                            OFFSET(%s - 1) * %s"
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "SELECT \
                g.id AS id, \
                g.id AS id2, \
                g.str_nome, \
                c.str_nome as nome_cliente, \
                g.bol_status \
            FROM grupos g \
            LEFT JOIN cliente c \
                ON c.id = g.fk_id_cliente \
            WHERE \
                g.id > 0 \
                %s %s %s %s \
            LIMIT %s \
            OFFSET(%s - 1) * %s"
            % (
                search_query_sql,
                template_query_sql,
                status_query_sql,
                order_query,
                limit,
                page,
                limit,
            ),
            (),
        )

    else:
        resposta = modulos.banco().sql(
            # "select g.id,g.str_nome, c.str_nome as nome_cliente,g.bol_status from grupos g left join cliente c on g.fk_id_cliente = c.id and fk_id_cliente=$1 \
            #                            %s %s %s %s LIMIT %s  OFFSET(%s - 1) * %s"
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "SELECT \
                g.id AS id, \
                g.id AS id2, \
                g.str_nome, \
                c.str_nome as nome_cliente, \
                g.bol_status \
            FROM grupos g \
            LEFT JOIN cliente c \
                ON g.fk_id_cliente = c.id AND fk_id_cliente = $1 \
            %s %s %s %s \
            LIMIT %s  \
            OFFSET(%s - 1) * %s"
            % (
                search_query_sql,
                template_query_sql,
                status_query_sql,
                order_query,
                limit,
                page,
                limit,
            ),
            (session["usuario"]["id_cliente"]),
        )

    resultado = {
        "meta": {
            "page": page,
            "pages": total_pages,
            "perpage": limit,
            "total": contagem_total,
            "sort": "asc",
            "field": "id",
        },
        "data": resposta,
        "aviso": {"texto": "Atenção: Caracteres inválidos no formulário!"}
        if query_perigoso
        else None,
    }

    return json.dumps(resultado)
