from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_exclui", methods=["POST"])
def _permissoes_exclui():
    """Rota para excluir a permissão

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuário, e redireciona caso não haja permissão
    3. Verifica se 'exclui_permissao' está salvo na sessão
    4. Obtém os dados do formulário
    5. Obtem os tokens, gerados no arquivo (_permissoes_verifica_exc.py)
    6. Verifica se os tokens são iguais
    7. Obtém o ID real da permissão
    8. Verifica se há usuários associados à permissão
    9. Exclui a permissão do 'une_grupo_a_modulos'
    10. Exclui a permissão do banco de dados
    11. Se excluído com sucesso, grava o log e retorna 0 (sucesso)

    Returns:
        Retorna: A resposta da exclusão da permissão
    """
    if request.method == "POST":
        # Verifica se o usuário está logado
        # if not 'usuario' in session.keys():
        # O uso de not in aprimora a leitura
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)

        # Obtem as permissoes do usuário, e retorna caso não haja permissão
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        if not menu[8][3]:
            retorno = {
                "status": "1",
                "msg": "Usuário não pode mais excluir permissões...",
            }
            return json.dumps(retorno)

        # Retorna se 'exclui_permissao' não estiver salvo na sessão
        if "exclui_permissao" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)
        else:
            # Obtem os dados do formulário
            form = request.form

            # Obtém os tokens, gerados no arquivo (_permissoes_verifica_exc.py)
            token = form.get("token")
            token_sys = session["exclui_permissao"]["token"]

            # Se forem iguais, exclui a permissão
            if int(token) == token_sys:
                # Obtém o real ID da permissão (Os tokens são para mascará-lo e proteger alterações não permitidas)
                id_exc = session["exclui_permissao"]["id"]

                # Verifica se há usuários utilizando a permissão
                resposta = modulos.banco().sql(
                    #"select id from usuario where fk_id_permissao=$1",
                    # Adicionando maiúsculas e aprimorando a sintaxe da query
                    "SELECT \
                        id \
                    FROM usuario \
                    WHERE \
                        fk_id_permissao = $1",
                    (id_exc),
                )
                # Se houver usuários utilizando a permissão, retorna o erro: Permissão em uso por usuários
                if resposta:
                    retorno = {"status": "1", "msg": "Permissão em uso por usuários..."}
                    return json.dumps(retorno)

                # Exclui a permissão do 'une_grupo_a_modulo'
                modulos.banco().sql(
                    # "delete from une_grupo_a_modulo where fk_id_grupo=$1 returning id",
                    # Adicionando maiúsculas e aprimorando a sintaxe da query
                    "DELETE FROM une_grupo_a_modulo \
                    WHERE \
                        fk_id_grupo = $1 \
                    RETURNING id",
                    (id_exc),
                )

                # Exclui a permissão do banco de dados
                resposta = modulos.banco().sql(
                    # "delete from grupos where id=$1 returning str_nome",
                    # Adicionando maiúsculas e aprimorando a sintaxe da query
                    "DELETE FROM grupos \
                    WHERE \
                        id = $1 \
                    RETURNING str_nome",
                    (id_exc),
                )

                # Se a permissão foi excluída, grava no log e retorna com sucesso
                if resposta:
                    session["msg"] = "Permissão excluída com sucesso"
                    session["msg_type"] = "success"
                    retorno = {"status": "0"}
                    modulos.log().log(
                        codigo_log=5,
                        ip=request.remote_addr,
                        modulo_log="PERMISSOES",
                        fk_id_usuario=session["usuario"]["id"],
                        texto_log="Permissão %s excluída" % (resposta[0]["str_nome"]),
                    )
                    return json.dumps(retorno)
            else:
                # Retorna o erro: Token incorreto
                retorno = {"status": "1", "msg": "Token incorreto..."}
                return json.dumps(retorno)
