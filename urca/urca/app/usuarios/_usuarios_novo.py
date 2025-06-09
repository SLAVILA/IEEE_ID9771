from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_novo", methods=["POST"])
def _usuarios_novo():
    """
    Cria um novo usuário no banco de dados.
    
    1. Verifica se o usuário estiver logado
    2. Verifica se possui as permissões necessárias
    3. Obtem os dados do formulário
    4. Verifica se os campos do formulário foram preenchidos corretamente, retornando erros se estiverem vazios ou errados
    5. Encripta a senha
    6. Cria o usuário no banco de dados

    Parâmetros:
        None

    Retorna:
        str: Uma string JSON contendo o status e a mensagem da operação.
            - Se o usuário não estiver logado.
            - Se o usuário não tiver permissão para criar usuários.
            - Se os campos de nome, email ou senha estiverem vazios.
            - Se a senha tiver menos de 4 caracteres.
            - Se o email for inválido.
            - Se a flag de admin for definida como "1" e nenhuma permissão ou cliente for selecionado.
            - Se o email já estiver cadastrado para outro usuário.
            - Se o usuário for criado com sucesso.
    """
    if request.method == "POST":
        # if not 'usuario' in session.keys():
        # O uso de not in aumenta a clareza ao ler a sintaxe
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)

        # Agora verifica se o usuário possui permissão para acessar os dados
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        if not menu[1][1]:
            retorno = {
                "status": "1",
                "msg": "Usuário não pode mais incluir usuários...",
            }
            return json.dumps(retorno)

        form = request.form
        str_nome = form.get("str_nome").strip()
        str_email = form.get("str_email").strip()
        str_senha = form.get("str_senha").strip()
        fk_id_permissao = form.get("fk_id_permissao")
        fk_id_cliente = form.get("fk_id_cliente")
        fk_id_setor = form.get("fk_id_setor")
        bol_status = form.get("bol_status")
        bol_admin = form.get("bol_admin")
        bol_trocou_senha = form.get("bol_trocou_senha")
        int_urna = form.get("int_urna")

        if len(str_nome) == 0:
            retorno = {
                "status": "1",
                "msg": "Nome não pode ser vazio...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        if len(str_email) == 0:
            retorno = {
                "status": "1",
                "msg": "Email não pode ser vazio...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        if len(str_senha) == 0:
            retorno = {
                "status": "1",
                "msg": "Senha não pode ser vazia...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        if len(str_senha) < 4:
            retorno = {
                "status": "1",
                "msg": "Senha deve conter no mínimo 4 caracteres...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        # if not "@" in str_email:
        if "@" not in str_email:
            retorno = {"status": "1", "msg": "Email inválido...", "msg_type": "danger"}
            return json.dumps(retorno)

        # print(fk_id_tipo)
        if bol_admin == "1":
            fk_id_permissao = None
        else:
            if len(fk_id_permissao) == 0:
                retorno = {
                    "status": "1",
                    "msg": "Selecione uma permissão para o usuário...",
                    "msg_type": "danger",
                }
                return json.dumps(retorno)

        if bol_admin == "1":
            fk_id_cliente = None
        else:
            if len(fk_id_cliente) == 0:
                retorno = {
                    "status": "1",
                    "msg": "Selecione um cliente para o usuário...",
                    "msg_type": "danger",
                }
                return json.dumps(retorno)

        resposta = modulos.banco().sql(
            # "select id from usuario where str_email=$1",
            # Adicionando maíusculas para aprimorar a sintaxe
            "SELECT id FROM usuario WHERE str_email = $1",
            (str_email),
        )
        if resposta:
            retorno = {
                "status": "1",
                "msg": "Email já cadastrado para outro usuario...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        if len(str_email) == 0:
            retorno = {
                "status": "1",
                "msg": "Email não pode ser vazio...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        # senha_enc=modulos.criptografia().encriptar(str_senha)
        senha_enc = hashlib.sha512(str_senha.encode()).hexdigest()

        resposta = modulos.banco().sql(
            # "insert into usuario (str_nome,str_email,str_senha, \
            # fk_id_permissao,bol_status,bol_admin,int_urna,fk_id_cliente,bol_trocou_senha,fk_id_setor) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) returning id",
            # Adicionando maíusculas para aprimorar a sintaxe
            "INSERT INTO usuario \
                (str_nome, str_email, str_senha, \
                fk_id_permissao, bol_status, bol_admin, \
                int_urna, fk_id_cliente, bol_trocou_senha, fk_id_setor) \
            VALUES \
                ($1, $2, $3, \
                $4, $5, $6, \
                $7, $8, $9, $10) \
            RETURNING id",
            (
                str_nome, # $1
                str_email, # $2
                senha_enc, # $3
                fk_id_permissao, # $4
                bol_status, # $5
                bol_admin, # $6
                int_urna, # $7
                fk_id_cliente, # $8
                bol_trocou_senha, # $9
                fk_id_setor, # $10
            ),
        )
        if resposta:
            # session['msg']='Usuário incluído com sucesso!'
            # session['msg_type']='success'
            retorno = {
                "status": "0",
                "msg": "Usuário %s incluído com sucesso!" % (str_nome),
            }
            # log = modulos.log().log(
            # Removendo o retorno do log, que não é utilizado
            modulos.log().log(
                codigo_log=5,
                ip=request.remote_addr,
                modulo_log="USUARIOS",
                fk_id_usuario=session["usuario"]["id"],
                fk_id_cliente=session["usuario"]["id_cliente"],
                texto_log="Criação de usuário. Login: %s, email: %s"
                % (str_nome, str_email),
            )
            return json.dumps(retorno)
