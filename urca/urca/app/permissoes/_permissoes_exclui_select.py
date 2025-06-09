from flask import request, session
from biblioteca import modulos
import json

from app.permissoes import b_permissoes


@b_permissoes.route("/_permissoes_exclui_select", methods=["POST"])
def _permissoes_exclui_select():
    """Rota excluir permissões selecionadas

    1. Verifica se o usuário está logado
    2. Obtém as permissoes do usuario, e retorna caso não possua permissão
    3. Obtem os dados do formulario
    4. Faz um loop excluindo cada permissão selecionado
    6. Grava o log e retorna a resposta

    Returns:
        Retorna: A resposta da exclusão das permissões
    """
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        retorno = {"status": "99"}
        return json.dumps(retorno)

    # Obtem as permissoes do usuario, e retorna caso não possua permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][3]:
        retorno = {"status": "1", "msg": "Usuário não pode mais excluir permissões..."}
        return json.dumps(retorno)
    else:
        # Obtem os dados do formulario
        form = request.form
        ids = form["ids"].replace("selec_", "").split("#")[1:]

        erros = []
        excluidos = 0

        # Faz um loop excluindo cada eleitor selecionado
        for i in ids:
            if i != "on":
                id_exc = i
                # Verifica se há usuários utilizando a permissão
                resposta = modulos.banco().sql(
                    # "select id from usuario where fk_id_permissao=$1",
                    # Adicionando maiúsculas e aprimorando a sintaxe da query
                    "SELECT \
                        usuario.id, \
                        grupos.str_nome \
                    FROM usuario \
                    LEFT JOIN grupos \
                        ON grupos.id = $1 \
                    WHERE \
                        fk_id_permissao = $1",
                    (id_exc),
                )
                # Se houver usuários utilizando a permissão, retorna o erro: Permissão em uso por usuários
                if resposta:
                    erros.append(
                        f"Permissão: {resposta[0]['str_nome']} em uso por usuários."
                    )
                    continue
                    # retorno = {"status": "1", "msg": "Permissão em uso por usuários..."}
                    # return json.dumps(retorno)

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

                # Grava o log
                if resposta:
                    excluidos += 1
                    modulos.log().log(
                        codigo_log=5,
                        ip=request.remote_addr,
                        modulo_log="PERMISSOES",
                        fk_id_usuario=session["usuario"]["id"],
                        texto_log="Exclusão de permissão. Permissão: %s"
                        % (resposta[0]["str_nome"]),
                    )
                    
    mensagem_erro = f"<br><br>Houve erros em {len(erros)} permissões: <br>{', '.join(f'{erro}<br>' for erro in erros)}" if erros else ''

    retorno = {
        "status": "0",
        "msg": f"Foram excluídas {excluidos} permissões com sucesso! {mensagem_erro}",
    }

    return json.dumps(retorno)
