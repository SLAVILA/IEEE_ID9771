from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
import sys, subprocess
import os
from config.config import CONFIG

from app.firewall import b_firewall


@b_firewall.route("/firewall", methods=["GET"])
def firewall():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("https://" + request.host + "login")

    # Obtem as permissoes do usuário e redireciona caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])

    if not menu[11][2]:
        session["msg"] = "Você não possui permissão para acessar este módulo..."
        session["msg_type"] = "danger"
        return redirect("https://" + request.host + "/")
    # CARREGA AS CONFIGURACOES DE CLIENTE DO BANCO

    # Obtém o IP do usuário
    ip = session["endereco_ip"]

    # Verifica se o firewall está ativo
    out = subprocess.Popen(
        ["/sbin/pfctl", "-s", "info"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, stderr = out.communicate()
    tmp = stdout.decode()[:-1].split("\n")[0]

    if "Enable" in tmp:
        ativo = True
    else:
        ativo = False

    # Obtem os IP's das exceções
    out = subprocess.Popen(
        ["/sbin/pfctl", "-t", "tabela_ssh", "-T", "show"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = out.communicate()
    excecoes = stdout.decode()[:-1].replace("   ", ",")

    # Renderiza a página do firewall
    return render_template(
        "firewall/firewall.html",
        usuario=session["usuario"],
        menu=menu,
        ip=ip,
        ativo=ativo,
        excecoes=excecoes,
    )


@b_firewall.route("/_firewall", methods=["POST"])
def _firewall():
    if request.method == "POST":
        # Verifica se o usuário está logado
        # if not 'usuario' in session.keys():
        # O uso de not in aprimora a leitura
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)
        
        # Obtem as permissoes do usuário
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        
        # Obtém os dados do formulário
        form = request.form
        
        # Obtém o IP escrito no formulário
        ip = form.get("ip").strip()
        if len(ip) == 0:
            retorno = {"status": "1", "msg": "Informe um endereço IP..."}
            return json.dumps(retorno)
        
        # Adiciona o IP na tabela de exceções
        os.system("/sbin/pfctl -t tabela_ssh -T add %s" % (ip))
        
        # Retorna o status
        retorno = {"status": "0", "msg": "IP adicionado com sucesso!"}
        return json.dumps(retorno)


@b_firewall.route("/_firewall_altera", methods=["POST"])
def _firewall_altera():
    if request.method == "POST":
        # Verifica se o usuário está logado
        # if not 'usuario' in session.keys():
        # O uso de not in aprimora a leitura
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)
        
        # Obtem as permissoes do usuário
        menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
        
        # Obtém os dados do formulário
        form = request.form
        ligado = form.get("ligado").strip()
        
        # Retorna o status, dependendo de se o firewall foi ligado ou desligado
        if int(ligado) == 1:
            os.system("/sbin/pfctl -e")
            retorno = {"status": "0", "msg": "Firewall ativado"}
        else:
            os.system("/sbin/pfctl -d")
            retorno = {"status": "0", "msg": "Firewall desativado"}

        return json.dumps(retorno)
