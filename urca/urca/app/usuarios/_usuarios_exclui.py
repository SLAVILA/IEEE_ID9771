from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_exclui", methods=["POST"])
def _usuarios_exclui():
    """
    Esta função lida com a solicitação POST para excluir um usuário.
    1.Verifica se o usuário está logado e tem a permissão necessária para excluir usuários.
    2. Se o usuário estiver logado e tiver permissão, verifica o token fornecido no formulário da solicitação.
    3. Se o token estiver correto, exclui o usuário do banco de dados e retorna um objeto JSON com um status de sucesso.

    Parâmetros:
        Nenhum

    Retorna:
        str: Um objeto JSON contendo o status da operação de exclusão.
    """
    if request.method == "POST":
        # Verifica se o usuário está logado
        # if not 'usuario' in session.keys():
        # O uso de not in aprimora a leitura
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)

        # Verifica se o usuário tem permissão para excluir usuários
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        if not menu[1][3]:
            retorno = {
                "status": "1",
                "msg": "Usuário não pode mais excluir usuários...",
            }
            return json.dumps(retorno)

        # if not "exclui_usuario" in session.keys():
        # O uso de not in aprimora a leitura
        if "exclui_usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)
        else:
            form = request.form
            token = form.get("token")
            token_sys = session["exclui_usuario"]["token"]
            print("# %s #" % (token_sys))
            print("# %s #" % (token))
            if int(token) == token_sys:
                # Se o token estiver correto
                print("aqui")
                id_exc = session["exclui_usuario"]["id"]
                resposta = modulos.banco().sql(
                    # "delete from usuario where id=$1 returning str_nome,str_email",
                    # Adicionando maíusculas e aprimorando a visualização
                    "DELETE \
                    FROM usuario \
                    WHERE \
                        id = $1 \
                    RETURNING str_nome, str_email",
                    (id_exc),
                )
                if resposta:
                    session["msg"] = "Usuário excluído com sucesso"
                    session["msg_type"] = "success"
                    retorno = {"status": "0"}

                    # log = modulos.log().log(
                    # Removendo o retorno do log pois não é utilizado
                    modulos.log().log(
                        codigo_log=5,
                        ip=request.remote_addr,
                        modulo_log="USUARIOS",
                        fk_id_usuario=session["usuario"]["id"],
                        texto_log="Exclusão de usuário. Login: %s, email: %s"
                        % (resposta[0]["str_nome"], resposta[0]["str_email"]),
                    )

                    return json.dumps(retorno)
            else:
                retorno = {"status": "1", "msg": "Token incorreto..."}
                # log = modulos.log().log(
                # Removendo o retorno do log pois não é utilizado
                modulos.log().log(
                    codigo_log=5,
                    ip=request.remote_addr,
                    modulo_log="USUARIOS",
                    fk_id_usuario=session["usuario"]["id"],
                    texto_log="Tentativa de manipulação de token de exclusão.",
                )

                return json.dumps(retorno)
