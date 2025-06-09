from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/permissoes_novo", methods=["GET"])
def permissoes_novo():
    """Rota para mostrar a página de adição de permissão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e redireciona caso não haja permissão
    3. Filtra pelos TEMPLATES, se estiverem na sessão
    4. Renderiza o formulário de adição de permissão

    Returns:
        Renderiza: O formulário de adição de permissão
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtem as permissoes do usuário, e redireciona caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][1]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Filtra pelos TEMPLATES, se estiverem na sessão
    if "lista_templates" in session.keys():
        if "not" in session["lista_templates"]:
            if session["usuario"]["admin"]:
                resposta = modulos.banco().sql(
                    # "SELECT id,str_nome FROM cliente WHERE bol_status AND NOT bol_template ORDER BY str_nome",
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
                resposta = modulos.banco().sql(
                    # "SELECT id,str_nome FROM cliente WHERE bol_status and bol_template ORDER BY str_nome",
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
        if session["usuario"]["admin"]:
            resposta = modulos.banco().sql(
                # "SELECT id,str_nome FROM cliente WHERE bol_status AND NOT bol_template ORDER BY str_nome",
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

    # clientes = resposta
    modulos_grupos = []
    modulos_dic = {}
    edita_grupo = []

    permissoes = modulos.banco().sql(
        # "select g.id,g.str_nome, c.str_nome as nome_cliente,g.bol_status from grupos g left join cliente c on c.id=g.fk_id_cliente where g.id > 0 %s %s %s %s LIMIT %s \
        #                            OFFSET(%s - 1) * %s"
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            g.id, \
            g.str_nome, \
            c.str_nome as nome_cliente, \
            g.bol_status \
        FROM grupos g \
        LEFT JOIN cliente c \
            ON g.fk_id_cliente = c.id AND fk_id_cliente = $1",
        (session["usuario"]["id_cliente"]),
    )
    
    # Renderiza
    return render_template(
        "permissoes/permissoes_edicao.html",
        usuario=session["usuario"],
        menu=menu,
        edita_grupo=edita_grupo,
        clientes=clientes,
        modulos=modulos_grupos,
        modulos_dic=modulos_dic,
        novo=1,
        permissoes=permissoes,
    )
