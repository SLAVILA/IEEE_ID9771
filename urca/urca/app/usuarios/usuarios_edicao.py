from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/usuarios_edicao", methods=["GET"])
def usuarios_edicao():
    """
    Renderiza o template para edição das informações do usuário.
    
    1. Verifica se o usuário estiver logado
    2. Verifica se possui as permissões necessárias
    3. Carrega o usuário para edição
    4. Carrega as permissões do usuário
    5. Carrega os clientes do usuário
    6. Carrega os setores do usuário
    7. Renderiza o template com as informações do usuário
    
    Retorna:
        O template renderizado para edição do usuário.
        
    Retorna:
        Redirect: Se a chave 'usuario' não estiver presente na sessão, redireciona para a página de login.
        Redirect: Se o usuário não tiver permissão para acessar os dados, redireciona para a página inicial.
    """
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")

    # Agora verifica se o usuário possui permissão para acessar os dados
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[1][2]:
        return redirect("https://" + request.host + "/")

    resposta = modulos.banco().sql(
        # "select u.id, u.str_nome,u.str_email,u.str_telefone,u.bol_trocou_senha,t.str_nome as str_nome_perm,t.id as id_perm,u.bol_status,bol_admin,u.int_urna,case when u.bol_status is true then \
        #'ATIVO' else 'INATIVO' end as str_status, c.str_nome as str_nome_cliente,c.id as id_cliente,u.usuario_tag1,u.usuario_tag2, u.usuario_tag3,u.usuario_tag4,u.fk_id_cliente,u.fk_id_setor from usuario u \
        # left join grupos t on u.fk_id_permissao=t.id \
        # left join cliente c on u.fk_id_cliente=c.id \
        # where u.id=$1;",
        # Adicionando maiúsculas para aprimorar a legibilidade
        "SELECT \
            u.id, \
            u.str_nome, \
            u.str_email, \
            u.str_telefone, \
            u.bol_trocou_senha, \
            t.str_nome AS str_nome_perm, \
            t.id AS id_perm, \
            u.bol_status, \
            bol_admin, \
            u.int_urna, \
            CASE WHEN u.bol_status is true THEN 'ATIVO' else 'INATIVO' END AS str_status, \
            c.str_nome AS str_nome_cliente, \
            c.id AS id_cliente, \
            u.usuario_tag1, \
            u.usuario_tag2, \
            u.usuario_tag3, \
            u.usuario_tag4, \
            u.fk_id_cliente, \
            u.fk_id_setor \
        FROM usuario u \
        LEFT JOIN grupos t \
            ON u.fk_id_permissao = t.id \
        LEFT JOIN cliente c \
            ON u.fk_id_cliente = c.id \
        WHERE \
            u.id = $1;",
        (session["edita_usuario"]),
    )

    if resposta:
        edita_usuario = resposta[0]

    # resposta = modulos.banco().sql(
    tipo_permissao = modulos.banco().sql(
        # "select id,str_nome as str_nome_perm from grupos order by str_nome",
        # Adicionando maiúsculas para aprimorar a legibilidade
        "SELECT \
            id, \
            str_nome AS str_nome_perm \
        FROM grupos \
        ORDER BY str_nome",
        (),
    )
    # tipo_permissao = resposta

    # resposta = modulos.banco().sql(
    # Alterando a variável para não utilizar uma variável 'resposta' toda vez
    clientes = modulos.banco().sql(
        # "select id,str_nome from cliente order by str_nome",
        # Adicionando maiúsculas para aprimorar a legibilidade
        "SELECT \
            id, \
            str_nome \
        FROM cliente \
        ORDER BY str_nome",
        (),
    )
    if session["usuario"]["admin"]:
        config_cliente = []
    else:
        config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    tmp = modulos.banco().sql(
        # Adicionando maiúsculas para aprimorar a legibilidade
        "SELECT \
            fk_id_setor \
        FROM usuario_setor \
        WHERE \
            fk_id_usuario = $1",
        (session["edita_usuario"]),
    )
    if not tmp:
        # setores_usuario = "0"
        # Variável não está sendo utilizada
        lista = []
    else:
        lista = []
        for i in tmp:
            lista.append(i["fk_id_setor"])

    setores=[]
            
    return render_template(
        "usuarios/usuarios_edicao.html",
        usuario=session["usuario"],
        menu=menu,
        edita_usuario=edita_usuario,
        tipo_permissao=tipo_permissao,
        setores_usuario=lista,
        setores=setores,
        clientes=clientes,
        novo=0,
        config_cliente=config_cliente,
    )
