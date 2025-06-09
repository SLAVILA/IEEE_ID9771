import json
from flask import jsonify, render_template, request, redirect, session, current_app
from biblioteca import modulos
from ast import literal_eval
from app.series_temporais import b_series_temporais


@b_series_temporais.route("/dados_individual", methods=["GET"])
def dados_individual():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    dados_obtidos = current_app.dados_obtidos.value

    if not dados_obtidos:
        session["msg"] = "Dados ainda não obtidos. Aguarde um instante."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Obtém as permissões do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[6][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    

    # CARREGA AS CONFIGURACOES DE CLIENTE DO BANCO
    config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    return render_template(
        "series_temporais/dados_individual.html",
        usuario=session["usuario"],
        menu=menu,
        config_cliente=config_cliente,
    )


@b_series_temporais.route("/dados_grupo", methods=["GET"])
def dados_grupo():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    dados_obtidos = current_app.dados_obtidos.value

    if not dados_obtidos:
        session["msg"] = "Dados ainda não obtidos. Aguarde um instante."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Obtém as permissões do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[7][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    

    # CARREGA AS CONFIGURACOES DE CLIENTE DO BANCO
    config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    return render_template(
        "series_temporais/dados_grupo.html",
        usuario=session["usuario"],
        menu=menu,
        config_cliente=config_cliente,
    )


@b_series_temporais.route("/dados_rollof", methods=["GET"])
def dados_rollof():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    dados_obtidos = current_app.dados_obtidos.value

    if not dados_obtidos:
        session["msg"] = "Dados ainda não obtidos. Aguarde um instante."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Obtém as permissões do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    

    # CARREGA AS CONFIGURACOES DE CLIENTE DO BANCO
    config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    return render_template(
        "series_temporais/dados_rollof.html",
        usuario=session["usuario"],
        menu=menu,
        config_cliente=config_cliente,
    )
    

@b_series_temporais.route("/visualizacao_dados_historicos", methods=["GET"])
def visualizacao_dados_historicos():
    if "usuario" not in session.keys():
        return redirect("login")
    
    dados_obtidos = current_app.dados_obtidos.value

    if not dados_obtidos:
        session["msg"] = "Dados ainda não obtidos. Aguarde um instante."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    # Obtém as permissões do usuário, e retorna caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    if not menu[8][0]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")

    

    # CARREGA AS CONFIGURACOES DE CLIENTE DO BANCO
    config_cliente = modulos.config().cliente(session["usuario"]["id_cliente"])

    return render_template(
        "series_temporais/outros_dados.html",
        usuario=session["usuario"],
        menu=menu,
        config_cliente=config_cliente,
    )