from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_verifica_exc", methods=["POST"])
def _permissoes_verifica_exc():
    """Rota para adicionar o ID da permissão selecionada à sessão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e redireciona caso não haja permissão
    3. Obtem os dados do formulário
    4. Verifica se a permissão está em uso por algum usuário
    5. Gera um token aleatório e salva na sessão
    6. Retorna a resposta, com o token

    Returns:
        Retorna: Sucesso (status 0), e o token
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    # Obtém as permissoes do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][3]:
        retorno = {"status": "1", "msg": "Usuário não pode mais excluir permissões..."}
        return json.dumps(retorno)

    # Obtem os dados do formulário
    form = request.form
    form_id = form.get("id")

    # Verifica se a permissão está em uso por algum usuário
    resposta = modulos.banco().sql(
        # "select id from usuario where fk_id_permissao=$1",
        # Adicionando maiúsculas e aprimorando a sintaxe da query
        "SELECT \
            id \
        FROM usuario \
        WHERE \
            fk_id_permissao = $1",
        (form_id),
    )
    print(request.form)
    if resposta:
        # Retorna com o erro: Permissão em uso por usuários
        retorno = {"status": "1", "msg": "Permissão em uso por usuários..."}
        return json.dumps(retorno)

    # Gera um token aleatório para mascarar o token real
    token = randint(10000000000, 99999999999)

    # Adiciona o ID da permissão a ser excluída, e o token, à sessão, e a marca como modificada
    session["exclui_permissao"] = {"id": form_id, "token": token}
    session.modified = True

    # Retorna com o token gerado
    retorno = {"status": "0", "token": token}
    return json.dumps(retorno)
