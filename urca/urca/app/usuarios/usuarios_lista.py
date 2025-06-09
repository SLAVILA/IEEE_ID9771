from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/usuarios_lista", methods=["GET"])
def usuarios_lista():
    """
    Uma função que trata a rota "/usuarios_lista".

    1. Verifica se o usuário estiver logado
    2. Verifica se o usuário possui as permissões necessárias
    4. Verifica se a chave 'nome_cliente' não estiver presente no dicionário 'usuario', redireciona para a página de login
    5. Verifica se a chave 'exclui_usuario' estiver presente nas chaves da sessão, removendo-a
    6. Verifica se a chave 'edita_usuario' estiver presente nas chaves da sessão, removendo-a
    7. Verifica se as chaves 'msg' e 'msg_type' estiverem presentes nas chaves da sessão, removendo-as
    8. Carrega uma lista de usuários, menu e clientes com base em certas condições
    9. Renderiza o template "usuarios/usuarios_lista.html" com as variáveis obtidas

    Parâmetros:
        Nenhum
    """
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")

    # Agora verifica se o usuário possui permissão para acessar os dados
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[1][0]:
        # Redireciona se não há permissão
        return redirect("https://" + request.host + "/")

    # if not 'nome_cliente' in session['usuario'].keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "nome_cliente" not in session["usuario"].keys():
        return redirect("https://" + request.host + "/login")

    if "exclui_usuario" in session.keys():
        session.pop("exclui_usuario")

    if "edita_usuario" in session.keys():
        session.pop("edita_usuario")

    usuarios = []

    if "msg" in session.keys():
        msg = session["msg"]
        msg_type = session["msg_type"]
        session.pop("msg")
        session.pop("msg_type")
    else:
        msg = ""
        msg_type = ""

    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])

    if "lista_templates" in session.keys():
        lista_templates, _ = modulos.query().sanitizar(
            session["lista_templates"]
        )
        
        tmp = modulos.banco().sql(
            # "select id,str_nome from cliente c where id > 0 %s order by str_nome"
            # Adicionando maiúsculas para aprimorar a leitura
            "SELECT \
                id, \
                str_nome \
            FROM cliente c \
            WHERE \
                id > 0 \
                %s \
            ORDER BY str_nome"
            % (lista_templates),
            (),
        )
    else:
        tmp = modulos.banco().sql(
            # "select id,str_nome from cliente where not bol_template order by str_nome",
            # Adicionando maiúsculas para aprimorar a leitura
            "SELECT \
                id, \
                str_nome \
            FROM cliente \
            WHERE \
                NOT bol_template \
            ORDER BY str_nome",
            (),
        )

    return render_template(
        "usuarios/usuarios_lista.html",
        usuario=session["usuario"],
        menu=menu,
        usuarios=usuarios,
        msg=msg,
        msg_type=msg_type,
        clientes=tmp,
    )
