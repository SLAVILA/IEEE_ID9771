from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math

from app.config_modulos import b_config_modulos


# dados para o datatable da lista 
@b_config_modulos.route("/_config_modulos_lista", methods=["POST"])
def _config_modulos_lista():
    """
    Método para carregar os módulos disponíveis ao usuário

    1. Verifica se o usuário está logado
    2. Verifica se o usuário tem permissão para acessar este módulo, caso não haja, redireciona para a tela inicial
    3. Obtém os dados do formulário
    4. Cria as variáveis e as define, junto com os queries
    5. Obtém a contagem dos módulos
    6. Obtém os módulos
    7. Retorna o resultado

    Returns:
        JSON: O resultado dos módulos
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")

    # Verifica se o usuário tem permissão para acessar este módulo, caso não haja, redireciona para a tela inicial
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[2][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")


    # Obtém os dados do formulário
    f = request.form

    # Cria as variáveis de paginacão, e as define, também gera queries programáticas
    page = 1
    limit = 10
    contagem_total = 0
    total_pages = 0
    sort_field = ""
    sort_ord = "asc"
    search_query = ""
    search_query_sql = ""
    order_query = ""


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
            search_query_sql = "WHERE str_nome_modulo ILIKE '%" + search_query + "%' "

    # Sanitiza os queries
    (
        (
            search_query_sql,  # Se tiver valor, filtra pelo termo de busca
            order_query,  # Se tiver valor, filtra pela ordem
            limit,  # Quantidade de registros por página
            page,  # Pagina sendo exibida
            limit,  # Quantidade de registros por página (cálculo do offset)
        ),
        query_perigoso,  # IMPORTANTE: QUERY PERIGOSO É GERADO QUANDO A QUERY POSSUI EXPRESSÕES PERIGOSAS!
    ) = modulos.query().sanitizar(
        search_query_sql,
        order_query,
        limit,
        page,
        limit,
    )

    # Obtém a contagem do total de módulos
    contagem = modulos.banco().sql(
        # "SELECT COUNT(id) \
        #                            FROM modulos \
        #                            %s "
        # Adicionando maiúsculas e modificando a query para aprimorar a legibilidade
        "SELECT \
            COUNT(id) \
        FROM modulos \
        %s "
        % (search_query_sql),
        (),
    )

    # print(" Contagem " + str(contagem))

    contagem_total = contagem[0]["count"]
    total_pages = math.floor(contagem_total / limit)

    # Obtém os módulos do banco de dados
    resposta = modulos.banco().sql(
        # "SELECT id, str_nome_modulo, str_descr_modulo FROM modulos \
        #                            %s %s LIMIT %s  OFFSET(%s - 1) * %s"
        # Adicionando maiúsculas e modificando a query para aprimorar a legibilidade
        "SELECT \
            id, \
            str_nome_modulo, \
            str_descr_modulo \
        FROM modulos \
        %s %s \
        LIMIT %s  OFFSET(%s - 1) * %s"
        % (search_query_sql, order_query, 
           limit, page, limit),
        (),
    )

    # print("Total de registros: " + str(contagem_total))

    # Retorna o resultado
    resultado = {
        "meta": {
            "page": page,
            "pages": total_pages,
            "perpage": limit,
            "total": contagem_total,
            "sort": "asc",
            "field": "id",
        },
        "data": [],
        "aviso": {"texto": "Atenção: Caracteres inválidos no formulário!"}
        if query_perigoso
        else None,
    }

    resultado["data"] = resposta

    return json.dumps(resultado)
