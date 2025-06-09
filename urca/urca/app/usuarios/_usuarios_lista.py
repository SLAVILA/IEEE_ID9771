from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_lista", methods=["POST"])
def _usuarios_lista():
    """
    Rota para recuperar a lista de usuários.

    1. Verifica se possui as permissões necessárias
    2. Inicializa as variáveis
    3. Cria queries programáticas para recuperar os dados, que se modificando conforme os parâmetros informados, e sanitiza
    4. Carrega a contagem dos usuários dependendo das TEMPLATES utilizadas.
    5. Carrega os usuários dependendo das TEMPLATES utilizadas.
    5. Cria o dicionário 'resposta' e retorna o JSON com os dados.

    Parâmetros:
    Nenhum parâmetro.

    Retorna:
        Redirect: Para a página principal, se não há permissões suficientes.
        JSON: Se executado com sucesso, um objeto JSON contendo a lista de usuários e metadados sobre a paginação.
    """
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")

    # Verifica se o usuário possui permissão para acessar os dados
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[1][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    f = request.form

    page = 1
    limit = 10
    contagem_total = 0
    total_pages = 0
    sort_field = ""
    sort_ord = "asc"
    search_query = ""
    search_query_sql = ""
    if "cliente_query_sql" in session.keys():
        cliente_query_sql = session["cliente_query_sql"]
    else:
        cliente_query_sql = ""
    order_query = ""
    status_query_sql = ""
    template_query = ""
    if "lista_templates" in session.keys():
        template_query_sql = session["lista_templates"]
    else:
        # template_query_sql = " AND not c.bol_template "
        # Adicionado maíusculas no query para facilitar a leitura
        # Caso não haja templates selecionados, define o padrão para SEM TEMPLATES
        template_query_sql = " AND NOT c.bol_template "

    # Obtém o número da página atual
    if "pagination[page]" in f.keys():
        page = int(f.get("pagination[page]"))

    # Obtém o número de itens por página (paginação)
    if "pagination[perpage]" in f.keys():
        limit = int(f.get("pagination[perpage]"))

    # Obtém o campo usado para ordenar
    if "sort[field]" in f.keys():
        
        sort_field = f.get("sort[field]")

    # Obtém o sentido da ordenação (asc ou desc)
    if "sort[sort]" in f.keys():
        sort_ord = f.get("sort[sort]")
        order_query = "ORDER BY " + sort_field + " " + sort_ord

    # Obtém o termo de busca da pesquisa
    if "query[generalSearch]" in f.keys():
        search_query = f.get("query[generalSearch]")
        if len(search_query) > 2:
            search_query_sql = " AND u.str_nome ILIKE '%" + search_query + "%' "

    # Obtém o cliente a ser filtrado
    if "query[Cliente]" in f.keys():
        id_cliente_sort = f.get("query[Cliente]")
        if int(id_cliente_sort) == 0:
            cliente_query_sql = "  "
        else:
            cliente_query_sql = " AND u.fk_id_cliente = %s " % (id_cliente_sort)
        session["cliente_query_sql"] = cliente_query_sql
        session["cliente_query_sql_id"] = int(id_cliente_sort)
        session.modified = True
        
        print("id_cliente_sort: ", id_cliente_sort)

    # Obtém o status a ser filtrado (0 = Todos, 1 = Ativos, 2 = Inativos)
    if "query[Status]" in f.keys():
        status_query = f.get("query[Status]")
        if status_query == "1":
            status_query_sql = " AND u.bol_status "
        elif status_query == "2":
            status_query_sql = " AND not u.bol_status  "

    # Obtém o template a ser filtrado (NÃO = 0, SIM = 1, MASTER = 2)
    if "query[Templates]" in f.keys():
        template_query = f.get("query[Templates]")
        if template_query == "0":
            template_query_sql = " AND not c.bol_template "
                
        elif template_query == "1":
            template_query_sql = " AND c.bol_template  "
        
        if template_query != session.get("lista_templates_query", None):
            cliente_query_sql = "  "
            id_cliente_sort = '0'
            session["cliente_query_sql"] = cliente_query_sql
            
        session["lista_templates_query"] = template_query
        session["lista_templates"] = template_query_sql
        
        session.modified = True
    else:
        session["cliente_query_sql_id"] = 0
        session["cliente_query_sql"] = "  "
        session["lista_templates_query"] = '0'
        session["lista_templates"] = " AND not c.bol_template "

    # Sanitiza os queries
    (
        (  # Necessário () pois é retornado um tuple, então quaisquer variáveis antes do query devem conter ()
            template_query_sql,  # Se tiver valor, filtra pelo template
            cliente_query_sql,  # Se tiver valor, filtra pelo cliente
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
        cliente_query_sql,
        search_query_sql,
        status_query_sql,
        order_query,
        limit,
        page,
        limit,
    )

    # if not template_query == "2":
    # Uso de != aumenta a clareza ao ler a sintaxe
    if template_query != "2":
        # Contagem TODOS os usuários utilizando das queries criadas anteriormente
        if session["usuario"]["admin"]:
            contagem = modulos.banco().sql(
                # "select count(u.id) from usuario u \
                # left join grupos t on t.id=u.fk_id_permissao \
                # left join cliente c on u.fk_id_cliente=c.id \
                # where c.id > 0 %s %s %s %s ;"
                # Adicionado maíusculas no query para facilitar a leitura
                "SELECT \
                    COUNT(u.id) \
                FROM usuario u \
                LEFT JOIN grupos t \
                    ON u.fk_id_permissao = t.id \
                LEFT JOIN cliente c \
                    ON u.fk_id_cliente = c.id \
                WHERE \
                    c.id > 0 \
                    %s %s %s %s ;"
                % (
                    template_query_sql,  # Se tiver valor, filtra pelo template
                    cliente_query_sql,  # Se tiver valor, filtra pelo cliente
                    search_query_sql,  # Se tiver valor, filtra pelo termo de busca
                    status_query_sql,  # Se tiver valor, filtra pelo status
                ),
                (),
            )
        else:
            contagem = modulos.banco().sql(
                # "select count(u.id) from usuario u, grupos t, cliente c where \
                # u.fk_id_permissao=t.id and u.fk_id_cliente=c.id and u.fk_id_cliente=$1 %s %s %s %s;"
                # Adicionado maíusculas no query para facilitar a leitura
                "SELECT \
                    COUNT(u.id) \
                FROM usuario u, grupos t, cliente c \
                WHERE \
                    u.fk_id_permissao = t.id \
                    AND u.fk_id_cliente = c.id \
                    AND u.fk_id_cliente = $1 \
                    %s %s %s %s;"
                % (
                    template_query_sql,  # Se tiver valor, filtra pelo template
                    cliente_query_sql,  # Se tiver valor, filtra pelo cliente
                    search_query_sql,  # Se tiver valor, filtra pelo termo de busca
                    status_query_sql,  # Se tiver valor, filtra pelo status
                ),
                (session["usuario"]["id_cliente"]),
            )
    else:
        # Conta apenas os usuários MASTER
        contagem = modulos.banco().sql(
            # "select count(u.id) from usuario u \
            # where u.bol_admin;",
            # Adicionado maíusculas no query para facilitar a leitura
            "SELECT \
                COUNT(u.id) \
            FROM usuario u \
            WHERE \
                u.bol_admin;",
            (),
        )

    # print(" Contagem " + str(contagem))

    contagem_total = contagem[0]["count"]
    total_pages = math.floor(contagem_total / limit)

    # if not template_query == "2":
    # Uso de != aumenta a clareza ao ler a sintaxe
    if template_query != "2":
        # Carrega todos os usuários utilizando das queries criadas anteriormente
        if session["usuario"]["admin"]:
            # resposta = modulos.banco().sql("select u.id, u.str_nome,u.str_email,u.str_telefone,t.str_nome as str_nome_perm, u.bol_status , \
            #                                case when (u.bol_admin) then 'SUPERADMIN' when (not u.bol_admin) then c.str_nome end as str_nome_cliente from usuario u, grupos t, cliente c where \
            #                               u.fk_id_permissao=t.id and u.fk_id_cliente=c.id %s %s %s LIMIT %s  OFFSET(%s - 1) * %s;" % (search_query_sql, status_query_sql,order_query, limit, page, limit),())
            resposta = modulos.banco().sql(
                "SELECT \
                    u.id as id, \
                    u.id as id2, \
                    u.str_nome, \
                    u.str_email, \
                    u.str_telefone, \
                    t.str_nome AS str_nome_perm, \
                    u.bol_status, \
                    CASE WHEN (u.bol_admin) THEN 'SUPERADMIN' WHEN (NOT u.bol_admin) THEN c.str_nome END AS str_nome_cliente \
                FROM usuario u \
                LEFT JOIN grupos t \
                    ON u.fk_id_permissao = t.id \
                LEFT JOIN cliente c \
                    ON u.fk_id_cliente = c.id \
                WHERE \
                    c.id > 0 \
                    %s %s %s %s %s \
                LIMIT %s  OFFSET(%s - 1) * %s "
                % (
                    template_query_sql,  # Se tiver valor, filtra pelo template
                    cliente_query_sql,  # Se tiver valor, filtra pelo cliente
                    search_query_sql,  # Se tiver valor, filtra pelo termo de busca
                    status_query_sql,  # Se tiver valor, filtra pelo status
                    order_query,  # Se tiver valor, filtra pela ordem
                    limit,  # Quantidade de registros por página
                    page,  # Pagina sendo exibida
                    limit,  # Quantidade de registros por página (cálculo do offset)
                ),
                (),
            )
        else:
            resposta = modulos.banco().sql(
                # "select u.id, u.str_nome,u.str_email,u.str_telefone,t.str_nome as str_nome_perm,u.bol_status, \
                # c.str_nome, u.int_urna as str_nome_cliente from usuario u, grupos t, cliente c where \
                # u.fk_id_permissao=t.id and u.fk_id_cliente=c.id and u.fk_id_cliente=$1 %s %s %s %s %s LIMIT %s  OFFSET(%s - 1) * %s;"
                # Adicionado maíusculas no query para facilitar a leitura
                "SELECT \
                    u.id as id, \
                    u.id as id2, \
                    u.str_nome, \
                    u.str_email, \
                    u.str_telefone, \
                    t.str_nome AS str_nome_perm, \
                    u.bol_status, \
                    c.str_nome, \
                    u.int_urna AS str_nome_cliente \
                FROM usuario u, grupos t, cliente c \
                WHERE \
                    u.fk_id_permissao = t.id \
                    AND u.fk_id_cliente = c.id \
                    AND u.fk_id_cliente = $1 \
                    %s %s %s %s %s \
                LIMIT %s  OFFSET(%s - 1) * %s;"
                % (
                    template_query_sql,  # Se tiver valor, filtra pelo template
                    cliente_query_sql,  # Se tiver valor, filtra pelo cliente
                    search_query_sql,  # Se tiver valor, filtra pelo termo de busca
                    status_query_sql,  # Se tiver valor, filtra pelo status
                    order_query,  # Se tiver valor, filtra pela ordem
                    limit,  # Quantidade de registros por página
                    page,  # Pagina sendo exibida
                    limit,  # Quantidade de registros por página (cálculo do offset)
                ),
                (session["usuario"]["id_cliente"]),
            )
    else:
        # Carrega apenas os usuários MASTER
        resposta = modulos.banco().sql(
            # "select \
            # u.id, u.str_nome,u.str_email,u.str_telefone, u.bol_status \
            # from usuario u where u.bol_admin",
            # Adicionado maíusculas no query para facilitar a leitura
            "SELECT \
                u.id, \
                u.str_nome, \
                u.str_email, \
                u.str_telefone, \
                u.bol_status \
            FROM usuario u \
            WHERE \
                u.bol_admin",
            (),
        )

    # print("Total de registros: " + str(contagem_total))
    
    tmp = modulos.banco().sql(
        # "select id,str_nome from cliente c where id > 0 %s order by str_nome"
        # Adicionando maiúsculas para aprimorar a leitura
        f"SELECT \
            id, \
            str_nome \
        FROM cliente c \
        WHERE \
            id > 0 \
            {'AND c.bol_template' if session.get('lista_templates_query', '') == '0' else 'AND not c.bol_template'} \
        ORDER BY str_nome",
        (),
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
        "clientes": tmp,
        "aviso": {"texto": "Atenção: Caracteres inválidos no formulário!"}
        if query_perigoso
        else None,
    }

    return json.dumps(resultado)
