from flask import request, session
from biblioteca import modulos
import json

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_exclui_select", methods=["POST"])
def _usuarios_exclui_select():
    """Rota excluir usuários selecionados

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuario, e retorna caso não possua permissão
    3. Obtem os dados do formulario
    4. Faz um loop excluindo cada usuário selecionado
    6. Grava o log e retorna a resposta

    Returns:
        Retorna: A resposta da exclusão dos usuários
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    # Obtem as permissoes do usuario, e retorna caso não possua permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[1][3]:
        retorno = {"status": "1", "msg": "Usuário não pode mais excluir permissões..."}
        return json.dumps(retorno)
    else:
        # Obtem os dados do formulario
        form = request.form
        ids = form["ids"].replace("selec_", "").split("#")[1:]

        erros = 0
        excluidos = 0

        # Faz um loop excluindo cada selecionado
        for i in ids:
            if i != "on":
                id_exc = i
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
                # Grava o log
                if resposta:
                    modulos.log().log(
                        codigo_log=5,
                        ip=request.remote_addr,
                        modulo_log="USUARIOS",
                        fk_id_usuario=session["usuario"]["id"],
                        texto_log="Exclusão de usuário. Login: %s, email: %s"
                        % (resposta[0]["str_nome"], resposta[0]["str_email"]),
                    )

                    excluidos += 1

                else:
                    erros += 1

    mensagem_erro = (
        f"<br><br>Houve erros em {erros} usuários."
        if erros > 0
        else ""
    )

    retorno = {
        "status": "0",
        "msg": f"Foram excluídos {excluidos} usuários com sucesso! {mensagem_erro}",
    }

    return json.dumps(retorno)
