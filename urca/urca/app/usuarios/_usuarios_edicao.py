from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math
import sys

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_edicao", methods=["POST"])
def _usuarios_edicao():
    """
    Atualiza as informações do usuário no banco de dados.

    1. Verifica se possui as permissões necessárias
    2. Obtém os dados do formulário
    3. Verifica se os dados do formulário são válidos
    4. Encripta a nova senha
    5. Atualiza as informações do usuário no banco de dados

    Parâmetros:
    - Nenhum

    Retorno:
    - Uma string JSON representando as informações atualizadas do usuário.
    - Se o usuário não estiver logado, retorna uma string JSON com o código de status 99.
    - Se o usuário não tiver as permissões necessárias, retorna uma string JSON com o código de status 1 e uma mensagem de erro.
    - Se os dados de entrada forem inválidos, retorna uma string JSON com o código de status 1 e uma mensagem de erro.
    - Se a atualização do usuário for bem-sucedida, retorna uma string JSON com o código de status 0 e uma mensagem de sucesso.
    """

    if request.method == "POST":
        # if not 'usuario' in session.keys():
        # O uso de not in aumenta a clareza ao ler a sintaxe
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)

        # Verifica se o usuário possui as permissões necessárias
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        if not menu[1][2]:
            retorno = {
                "status": "1",
                "msg": "Usuário não pode mais alterar usuários...",
            }
            return json.dumps(retorno)

        form = request.form
        str_nome = form.get("str_nome").strip()
        str_email = form.get("str_email").strip()
        str_senha = form.get("str_senha").strip()
        fk_id_permissao = form.get("fk_id_permissao")
        fk_id_cliente = form.get("fk_id_cliente")
        bol_status = form.get("bol_status")
        fk_id_setor = form.get("fk_id_setor")
        bol_admin = form.get("bol_admin")
        bol_trocou_senha = form.get("bol_trocou_senha")
        usuario_tag1 = form.get("usuario_tag1")
        usuario_tag2 = form.get("usuario_tag2")
        usuario_tag3 = form.get("usuario_tag3")
        usuario_tag4 = form.get("usuario_tag4")
        int_urna = form.get("int_urna")
        if bol_admin == "1":
            fk_id_cliente = None
            fk_id_permissao = None

        if len(fk_id_setor) == 0:
            retorno = {
                "status": "1",
                "msg": "Selecione um setor...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)
        # if int(fk_id_setor) == '0':
        #    fk

        lista_setor = fk_id_setor.split(",")
        if "0" in lista_setor:
            fk_id_setor = 0
        else:
            fk_id_setor = lista_setor[0]

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

        if len(str_senha) > 0 and len(str_senha) < 4:
            retorno = {
                "status": "1",
                "msg": "Senha deve conter no mínimo 4 caracteres...",
                "msg_type": "danger",
            }
            return json.dumps(retorno)

        # f not "@" in str_email:
        # O uso de not in aumenta a clareza ao ler a sintaxe
        if "@" not in str_email:
            retorno = {"status": "1", "msg": "Email inválido...", "msg_type": "danger"}
            return json.dumps(retorno)

        resposta = modulos.banco().sql(
            # "select id from usuario where not id=$1 and str_email=$2",
            # Adicionando maiúsculas para aprimorar a legibilidade
            "SELECT \
                id \
            FROM usuario \
            WHERE \
                NOT id = $1 \
                AND str_email = $2",
            (session["edita_usuario"], str_email),
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

        if len(str_senha) > 0:
            # senha_enc=modulos.criptografia().encriptar(str_senha)
            senha_enc = hashlib.sha512(str_senha.encode()).hexdigest()
            resposta = modulos.banco().sql(
                # "update usuario set str_nome=$1,str_email=$2,str_senha=$3, \
                # fk_id_permissao=$4,fk_id_cliente=$5,bol_status=$6,bol_admin=$7, usuario_tag1=$8, usuario_tag2=$9, usuario_tag3=$10,\
                # usuario_tag4=$11,int_urna=$12,bol_trocou_senha=$14,fk_id_setor=$15 where id=$13 returning id",
                # Adicionando maiúsculas para aprimorar a legibilidade
                "UPDATE usuario \
                SET \
                    str_nome = $1, \
                    str_email = $2, \
                    str_senha = $3, \
                    fk_id_permissao = $4, \
                    fk_id_cliente = $5, \
                    bol_status = $6, \
                    bol_admin = $7, \
                    usuario_tag1 = $8, \
                    usuario_tag2 = $9, \
                    usuario_tag3 = $10, \
                    usuario_tag4 = $11, \
                    int_urna = $12, \
                    bol_trocou_senha = $14, \
                    fk_id_setor = $15 \
                WHERE \
                    id = $13 \
                RETURNING id",
                (
                    str_nome,  # $1
                    str_email,  # $2
                    senha_enc,  # $3
                    fk_id_permissao,  # $4
                    fk_id_cliente,  # $5
                    bol_status,  # $6
                    bol_admin,  # $7
                    usuario_tag1,  # $8
                    usuario_tag2,  # $9
                    usuario_tag3,  # $10
                    usuario_tag4,  # $11
                    int_urna,  # $12
                    session["edita_usuario"],  # $13
                    bol_trocou_senha,  # $14
                    fk_id_setor,  # $15
                ),
            )
            # log = modulos.log().log(
            # Removendo o retorno do log, que não está sendo utilizado
            modulos.log().log(
                codigo_log=5,
                ip=request.remote_addr,
                modulo_log="USUARIOS",
                fk_id_usuario=session["usuario"]["id"],
                fk_id_cliente=fk_id_cliente,
                texto_log="Alteração de usuário com alteração de senha. Login: %s, email: %s"
                % (str_nome, str_email),
            )
        else:
            resposta = modulos.banco().sql(
                # "update usuario set str_nome=$1,str_email=$2, \
                # fk_id_permissao=$3,fk_id_cliente=$4,bol_status=$5,bol_admin=$6,usuario_tag1=$7, usuario_tag2=$8, usuario_tag3=$9,usuario_tag4=$10, int_urna=$11,\
                # bol_trocou_senha=$13,fk_id_setor=$14 where id=$12 returning id",
                # Adicionando maiúsculas para aprimorar a legibilidade
                "UPDATE usuario \
                SET \
                    str_nome = $1, \
                    str_email = $2, \
                    fk_id_permissao = $3, \
                    fk_id_cliente = $4, \
                    bol_status = $5, \
                    bol_admin = $6, \
                    usuario_tag1 = $7, \
                    usuario_tag2 = $8, \
                    usuario_tag3 = $9, \
                    usuario_tag4 = $10, \
                    int_urna = $11, \
                    bol_trocou_senha = $13, \
                    fk_id_setor = $14 \
                WHERE \
                    id = $12 \
                RETURNING id",
                (
                    str_nome,  # $1
                    str_email,  # $2
                    fk_id_permissao,  # $3
                    fk_id_cliente,  # $4
                    bol_status,  # $5
                    bol_admin,  # $6
                    usuario_tag1,  # $7
                    usuario_tag2,  # $8
                    usuario_tag3,  # $9
                    usuario_tag4,  # $10
                    int_urna,  # $11
                    session["edita_usuario"],  # $12
                    bol_trocou_senha,  # $13
                    fk_id_setor,  # $14
                ),
            )
            # log = modulos.log().log(
            # Removendo o retorno do log, que não está sendo utilizado
            modulos.log().log(
                codigo_log=5,
                ip=request.remote_addr,
                modulo_log="USUARIOS",
                fk_id_usuario=session["usuario"]["id"],
                fk_id_cliente=fk_id_cliente,
                texto_log="Alteração de usuário sem alteração de senha. Login: %s, email: %s"
                % (str_nome, str_email),
            )

        # tmp = modulos.banco().sql(
        # Removendo o retorno do query, que não está sendo utilizado
        modulos.banco().sql(
            # "delete from usuario_setor where fk_id_usuario=$1 returning id",
            # Adicionando maiúsculas para aprimorar a legibilidade
            "DELETE FROM usuario_setor \
            WHERE \
                fk_id_usuario = $1 \
            RETURNING id",
            (session["edita_usuario"]),
        )
        # if not "0" in lista_setor:
        # O uso de not in aumenta a clareza ao ler a sintaxe
        if "0" not in lista_setor:
            for i in lista_setor:
                # tmp = modulos.banco().sql(
                # Removendo o retorno do query, que não está sendo utilizado
                modulos.banco().sql(
                    # "insert into usuario_setor (fk_id_usuario,fk_id_setor,fk_id_cliente) values ($1,$2,$3) returning id",
                    # Adicionando maiúsculas para aprimorar a legibilidade
                    "INSERT INTO usuario_setor \
                        (fk_id_usuario, fk_id_setor, fk_id_cliente) \
                    VALUES \
                        ($1, $2, $3) \
                    RETURNING id",
                    (session["edita_usuario"], i, fk_id_cliente),
                )

        if resposta:
            # session['msg']='Usuário alterado com sucesso!'
            # session['msg_type']='success'
            retorno = {"status": "0", "msg": "Usuário alterado com sucesso!"}
            return json.dumps(retorno)
