from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_altera_nome", methods=["POST"])
def _permissoes_altera_nome():
    """Rota para alterar o nome da permissão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e redireciona caso não haja permissão
    3. Obtem os dados do formulário
    4. Verifica se o nome da permissão existe, e retorna caso exista
    5. Altera o nome da permissão
    6. Retorna a resposta
    
    Returns:
        Retorna: Status 0 (sucesso)
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")

    # Obtem as permissoes do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][2]:
        retorno = {"status": "1", "msg": "Usuário não pode mais alterar permissões..."}
        return json.dumps(retorno)

    # Obtem os dados do formulário
    form = request.form
    nome = form.get("str_nome")
    grupo = form.get("grupo")
    fk_id_cliente = form.get("fk_id_cliente")

    # Verifica se o nome da permissão existe, e retorna caso exista
    resposta = modulos.banco().sql(
        # "select id from grupos where str_nome=$1 and not id=$2",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            id \
        FROM grupos \
        WHERE \
            str_nome = $1 AND \
            NOT id = $2",
        (nome, grupo),
    )
    if resposta:
        retorno = {
            "status": "1",
            "msg": "Permissão com o mesmo nome já cadastrada",
            "msg_type": "danger",
        }
        return json.dumps(retorno)

    # Este tipo de query retorna o valor antigo, e se quiser, o novo também (x.str_nome)
    if len(fk_id_cliente) > 0:
        # Atualiza o nome da permissão no banco de dados
        resposta = modulos.banco().sql(
            # "update grupos x set str_nome=$1,fk_id_cliente=$2 from grupos y where x.id=y.id and x.id=$3 returning y.str_nome",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE grupos x \
            SET \
                str_nome = $1, \
                fk_id_cliente = $2 \
            FROM grupos y \
            WHERE \
                x.id = y.id AND \
                x.id = $3 \
            RETURNING y.str_nome",
            (nome, fk_id_cliente, grupo),
        )
    else:
        # Atualiza o nome da permissão no banco de dados
        resposta = modulos.banco().sql(
            # "update grupos x set str_nome=$1 from grupos y where x.id=y.id and x.id=$2 returning y.str_nome",
            # Adicionando maiúsculas e aprimorando a sintaxe da query
            "UPDATE grupos x \
            SET \
                str_nome = $1 \
            FROM grupos y \
            WHERE \
                x.id = y.id AND \
                x.id = $2 \
            RETURNING y.str_nome",
            (nome, grupo),
        )

    # Loga o evento
    modulos.log().log(
        codigo_log=5,
        ip=request.remote_addr,
        modulo_log="PERMISSOES",
        fk_id_usuario=session["usuario"]["id"],
        texto_log="Nome da permissão %s alterado para %s"
        % (resposta[0]["str_nome"], nome),
    )

    # Retorna os dados
    retorno = {"status": "0"}
    return json.dumps(retorno)
