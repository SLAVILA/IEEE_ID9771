from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_enviar_email", methods=["POST"])
def _usuarios_enviar_email():
    """Envia um email de recuperação de senha para o usuário informado
    
    1. Obtém os dados do formulário.
    2. Verifica se o usuário existe.
    3. Carrega a configuração do cliente.
    4. Envia um email de recuperação de senha para o usuário informado.
    5. Retorna um objeto JSON com um status de sucesso.
    
    Parâmetros:
        Nenhum
        
    Retorna:
        str: Um objeto JSON contendo o status da operação.
    """
    
    form = request.form
    usuario_id = form.get("id")
    
    # Obtém o usuário
    rows_usuarios = modulos.banco().sql(
        # "select * from usuario where  id = $1 order by str_nome"
        # Adicionando maiúsculas ao query, e aprimorando sua visualização
        "SELECT \
            * \
        FROM usuario \
        WHERE \
            id = $1 \
        ORDER BY str_nome",
        (usuario_id),
    )
    """
    if rows_usuarios:
        rows_usuarios = rows_usuarios
    else:
        rows_usuarios = []
    """
    # Comentado o código acima, aprimorando-o abaixo
    if not rows_usuarios:
        rows_usuarios = []
        
    # Carrega a configuração do cliente
    if session["usuario"]["admin"]:
        tmp = modulos.banco().sql(
            # "select fk_id_cliente from usuario where id=$1",
            # Adicionando maiúsculas ao query, e aprimorando sua visualização
            "SELECT \
                fk_id_cliente \
            FROM usuario \
            WHERE \
                id = $1",
            (usuario_id),
        )

        config_cliente = modulos.config().cliente(tmp[0]["fk_id_cliente"])
    else:
        config_cliente = modulos.config().cliente(rows_usuarios[0]["fk_id_cliente"])
    
    # Gera a nova senha e encripta-a
    senha_gerada = modulos.senha().gerar()
    senha_gerada_encriptada = hashlib.sha512(senha_gerada.encode()).hexdigest()

    #resposta_update = modulos.banco().sql(
    # Removendo o retorno do query pois não é usado
    # Faz o update da nova senha, já encriptada, no banco de dados
    modulos.banco().sql(
        # "update usuario set str_senha = $1, bol_trocou_senha='f' where id = $2 returning id",
        # Adicionando maiúsculas ao query, e aprimorando sua visualização
        "UPDATE usuario \
        SET \
            str_senha = $1, \
            bol_trocou_senha='f' \
        WHERE \
            id = $2 \
        RETURNING id",
        (senha_gerada_encriptada, usuario_id),
    )

    # ENVIO_EMAIL = None
    # Variável acima não está sendo usada
    
    # Gera o e-mail de envio de senha
    if session["usuario"]["admin"]:
        ASSUNTO_EMAIL_TEXTO = "URCA TRADING - Senha de Acesso"
        ENVIO_EMAIL_TEXTO = """<p>Prezado,</p>
        <p>Segue dados para acessar o painel administrativo do sistema URCA TRADING.</p>
        <p>Link: <a href="https://%s">https://%s</a></p>
        <p>Login: seu email</p>
        <p>Senha: <strong>{senha}</strong></p>
        <p>Cordialmente,</p>
        <p>URCA TRADING</p>
        """ % (
            request.host,
            request.host,
        )

    else:
        ENVIO_EMAIL_TEXTO = config_cliente["texto_email_envio_senha_admin"]
        ASSUNTO_EMAIL_TEXTO = config_cliente["assunto_email_envio_senha_admin"]

    texto = ENVIO_EMAIL_TEXTO.replace("{senha}", senha_gerada)

    para = rows_usuarios[0]["str_email"]

    # Alterna entre enviar por gmail, aws, zimbra, iagente e lk6 dependendo da configuração, e envia o e-mail
    if session["usuario"]["admin"]:
        tmp = modulos.email().smtp('services@urcatrading.com', para, ASSUNTO_EMAIL_TEXTO, texto)
    
    
    if tmp["status"] == "0":
        # Email enviado
        msg = "Senha enviada por e-mail."
        msg_type = "success"
        retorno = {"status": "0", "msg": msg, "msg_type": msg_type}
        # log = modulos.log().log(
        # Removendo o retorno do log, pois não é utilizado
        modulos.log().log(
            codigo_log=5,
            ip=request.remote_addr,
            modulo_log="USUARIOS",
            fk_id_usuario=session["usuario"]["id"],
            texto_log="Senha enviada por email para login %s."
            % (rows_usuarios[0]["str_email"]),
        )
    else:
        # Falha ao enviar o e-mail
        msg = tmp["msg"]
        msg_type = "danger"
        # log = modulos.log().log(
        # Removendo o retorno do log, pois não é utilizado
        modulos.log().log(
            codigo_log=5,
            ip=request.remote_addr,
            modulo_log="USUARIOS",
            fk_id_usuario=session["usuario"]["id"],
            texto_log="Falha ao enviar senha por email para login %s."
            % (rows_usuarios[0]["str_email"]),
        )
        retorno = {"status": "1", "msg": msg, "msg_type": msg_type}

    return json.dumps(retorno)
