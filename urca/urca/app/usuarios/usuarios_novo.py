from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/usuarios_novo", methods=["GET"])
def usuarios_novo():
    """
    Rota para criar um novo usuário.
    
    1. Verifica se o usuário estiver logado
    2. Verifica se o usuário possui as permissões necessárias
    3. Carrega o tipo de permissão do usuário
    4. Carrega os clientes do usuário
    5. Carrega os setores do usuário
    6. Carrega as configurações do usuário
    7. Renderiza o template 'usuarios/usuarios_novo.html' com as informações obtidas

    Retorna:
        - Se o usuário não estiver logado, redireciona para a página de login.
        - Se o usuário não tiver as permissões necessárias, redireciona para a página inicial.
        - Se o usuário for administrador, recupera uma lista de tipos de permissão do banco de dados.
        - Se o usuário não for administrador, recupera uma lista de tipos de permissão específica para seu cliente do banco de dados.
        - Renderiza o template 'usuarios/usuarios_edicao.html'
    """
    # if not 'usuario' in session.keys():
    # O uso de not in aumenta a clareza ao ler a sintaxe
    if "usuario" not in session.keys():
        return redirect("login")
    
    # Agora verifica se o usuário possui permissão para acessar os dados
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[1][1]:
        # Redireciona se não há permissão
        return redirect("https://" + request.host + "/")

    edita_usuario = {}
    print(session)
    
    # Carrega o tipo de permissão do usuário
    if session["usuario"]["admin"]:
        # resposta = modulos.banco().sql(
        # Alterado o nome da variável, desnecessário o uso de 'resposta' como variável extra
        tipo_permissao = modulos.banco().sql(
            # "select id,str_nome as str_nome_perm from grupos order by str_nome", ()
            # Adicionando maíusculas para melhorar a leitura da sintaxe
            "SELECT \
                id, \
                str_nome AS str_nome_perm \
            FROM grupos \
            ORDER BY str_nome",
            (),
        )
    else:
        # resposta = modulos.banco().sql(
        # Alterado o nome da variável, desnecessário o uso de 'resposta' como variável extra
        tipo_permissao = modulos.banco().sql(
            # "select id,str_nome as str_nome_perm from grupos where fk_id_cliente=$1 order by str_nome",
            # Adicionando maíusculas para melhorar a leitura da sintaxe
            "SELECT \
                id, \
                str_nome AS str_nome_perm \
            FROM grupos \
            WHERE \
                fk_id_cliente = $1 \
            ORDER BY str_nome",
            (session["usuario"]["id_cliente"]),
        )
    # tipo_permissao = resposta
    
    # Carrega os clientes do usuário
    if session["usuario"]["admin"]:
        # resposta = modulos.banco().sql(
        # Alterado o nome da variável, desnecessário o uso de 'resposta' como variável extra
        clientes = modulos.banco().sql(
            # "select id,str_nome from cliente order by str_nome",
            # Adicionando maíusculas para melhorar a leitura da sintaxe
            "SELECT \
                id, \
                str_nome \
            FROM cliente \
            ORDER BY str_nome",
            (),
        )
    else:
        # resposta = modulos.banco().sql(
        # Alterado o nome da variável, desnecessário o uso de 'resposta' como variável extra
        clientes = modulos.banco().sql(
            # "select id,str_nome from cliente where id=$1",
            # Adicionando maíusculas para melhorar a leitura da sintaxe
            "SELECT \
                id, \
                str_nome \
            FROM cliente \
            WHERE \
                id = $1",
            (session["usuario"]["id_cliente"]),
        )
    # clientes = resposta
    
    # Carrega os setores do usuário
    # resposta = modulos.banco().sql(
    # Alterado o nome da variável, desnecessário o uso de 'resposta' como variável extra
    setores = []
    
    # Carrega as configurações do cliente
    if session["usuario"]["admin"]:
        config_cliente = []
    else:
        # confiig_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])
        # Corrigido o nome da variável. Em meus testes, um usuário normal não consegue recuperar a configuração do cliente
        # por causa do 'ii' duplo no nome da variável
        # TODO: É um comportamento esperado? Se sim, ignorar a correção
        config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    return render_template(
        "usuarios/usuarios_edicao.html",
        usuario=session["usuario"],
        menu=menu,
        setores=setores,
        edita_usuario=edita_usuario,
        tipo_permissao=tipo_permissao,
        clientes=clientes,
        novo=1,
        config_cliente=config_cliente,
    )
