from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
import sys, subprocess
import os
from config.config import CONFIG
import psutil

from app.servidor import b_servidor

def convert_bytes(size, formato='KB'):
    # Definir os possíveis formatos
    formatos = ['B', 'KB', 'MB', 'GB', 'TB']

    # Iniciar com o menor formato
    index = 0

    # Converter bytes para o formato especificado
    while size > 1024 and index < len(formatos) - 1:
        size /= 1024.0
        index += 1

    return "{:.1f} {}".format(size, formatos[index])

@b_servidor.route("/uso_server", methods=["GET"])
def servidor():
    # Verifica se o usuário está logado
    if "usuario" not in session.keys():
        return redirect("https://" + request.host + "login")

    # Obtem as permissoes do usuário e redireciona caso não haja permissão
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    # Obter o uso de disco com psutil
    usage = psutil.disk_usage("/")

    # Obter o uso de CPU com psutil
    cpu_usage = psutil.cpu_percent()

    # Obter o uso de memória com psutil
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total  # Convertendo bytes para gigabytes
    used_memory_gb = memory.total - memory.free    # Convertendo bytes para gigabytes

    # Obter o uso de rede com psutil
    net_io = psutil.net_io_counters()
    network_usage = (net_io.bytes_recv + net_io.bytes_sent)
    
    dados = {
        "uso_disco": convert_bytes(usage.total - usage.free),
        "espaco_total_disco": convert_bytes(usage.total),
        "porcentagem_disco": usage.percent,
        "uso_cpu": cpu_usage,
        "uso_memoria": used_memory_gb,
        "total_memoria": total_memory_gb,
        "uso_rede": convert_bytes(network_usage),
    }

    # Renderiza a página do firewall
    return render_template(
        "servidor/status_maquina.html",
        usuario=session["usuario"],
        menu=menu,
        dados=dados
    )

@b_servidor.route("/_status_server", methods=["POST"])
def _status_server():
    if request.method == "POST":
        # Verifica se o usuário está logado
        if "usuario" not in session.keys():
            retorno = {"status": "99"}
            return json.dumps(retorno)
        
        # Obter o uso de disco com psutil
        usage = psutil.disk_usage("/")

        # Obter o uso de CPU com psutil
        cpu_usage = psutil.cpu_percent()

        # Obter o uso de memória com psutil
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total  # Convertendo bytes para gigabytes
        used_memory_gb = memory.total - memory.free    # Convertendo bytes para gigabytes

        # Obter o uso de rede com psutil
        net_io = psutil.net_io_counters()
        network_usage = (net_io.bytes_recv + net_io.bytes_sent)
        
        retorno = {
            "status": "0",
            "msg": "OK",
            "uso_disco": convert_bytes(usage.total - usage.free),
            "espaco_total_disco": convert_bytes(usage.total),
            "porcentagem_disco": usage.percent,
            "uso_cpu": cpu_usage,
            "uso_memoria": used_memory_gb,
            "total_memoria": total_memory_gb,
            "uso_rede": convert_bytes(network_usage),
        }
        return json.dumps(retorno)
