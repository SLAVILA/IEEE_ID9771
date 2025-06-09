from datetime import timedelta
import os
from flask import render_template, request, session, redirect, current_app
from biblioteca import modulos
import json
import math
from ast import literal_eval
import pandas as pd
from biblioteca.modulos import banco
from app.processos import b_processos

def format_duracao(duracao):
    segundos = int(duracao.total_seconds())
    
    if segundos < 60:
        return f"{segundos} segundos"
    elif segundos < 3600:
        minutos = segundos // 60
        segundos_restantes = segundos % 60
        if minutos == 1:
            if segundos_restantes == 0:
                return f"{minutos} minuto"
            return f"{minutos} minuto e {segundos_restantes} segundos"
        return f"{minutos} minutos" + (f" e {segundos_restantes} segundos" if segundos_restantes > 0 else "")
    elif segundos < 86400:
        horas = segundos // 3600
        minutos_restantes = (segundos % 3600) // 60
        segundos_restantes = segundos % 60
        if horas == 1:
            if minutos_restantes == 0 and segundos_restantes == 0:
                return f"{horas} hora"
            return f"{horas} hora" + (f", {minutos_restantes} minutos" if minutos_restantes > 0 else "") + (f" e {segundos_restantes} segundos" if segundos_restantes > 0 else "")
        return f"{horas} horas" + (f", {minutos_restantes} minutos" if minutos_restantes > 0 else "") + (f" e {segundos_restantes} segundos" if segundos_restantes > 0 else "")
    else:
        dias = segundos // 86400
        horas_restantes = (segundos % 86400) // 3600
        minutos_restantes = (segundos % 3600) // 60
        segundos_restantes = segundos % 60
        if dias == 1:
            if horas_restantes == 0 and minutos_restantes == 0 and segundos_restantes == 0:
                return f"{dias} dia"
            return f"{dias} dia" + (f", {horas_restantes} horas" if horas_restantes > 0 else "") + (f", {minutos_restantes} minutos" if minutos_restantes > 0 else "") + (f" e {segundos_restantes} segundos" if segundos_restantes > 0 else "")
        return f"{dias} dias" + (f", {horas_restantes} horas" if horas_restantes > 0 else "") + (f", {minutos_restantes} minutos" if minutos_restantes > 0 else "") + (f" e {segundos_restantes} segundos" if segundos_restantes > 0 else "")

    
nome_map = {
    "atualizar_banco": "Atualizar Banco de Dados",
    "atualizar_dados": "Atualizar Banco de Dados",
    "iniciar_markov": "Iniciar Markov",
    "iniciar_estocastico": "Iniciar Estocástico",
    "iniciar_analise": "Análise de Backtest",
    "iniciar_longo_prazo": "Iniciar Longo Prazo"
}

@b_processos.route("/processos", methods=["GET"])
def processos():
    # Verificação de usuário na sessão
    if "usuario" not in session.keys():
        return redirect("login")
    
    if not session['usuario'].get('admin', None):
        return redirect("/")
    
    # Definir número de processos por página e pegar o número da página da URL
    processos_por_pagina = 10
    pagina = request.args.get('page', 1, type=int)
    offset = (pagina - 1) * processos_por_pagina

    try:
        # Query para pegar processos com paginação
        query = """
            SELECT nome, pid_processo, data_inicio, id_usuario
            FROM processo
            ORDER BY data_inicio DESC
        """
        resultados = banco().sql(query, ())
        
        processos = []
        for resultado in resultados:
            processo = {
                "nome": nome_map[resultado["nome"]],
                "pid_processo": resultado["pid_processo"],
                "data_inicio": (resultado["data_inicio"] - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M:%S'),
                "id_usuario": resultado["id_usuario"]
            }
            processos.append(processo)
        
    except Exception as e:
        print(f"Erro ao obter processos: {e}")
        processos = []
    
    try:
        # Query para pegar processos completos com paginação
        query = """
            SELECT nome, data_inicio, data_fim, duracao, status
            FROM processos_completos
            WHERE EXTRACT(EPOCH FROM duracao) > 10
            ORDER BY data_inicio DESC
            LIMIT $1 OFFSET $2
        """
        resultados = banco().sql(query, (processos_por_pagina, offset))
        
        processos_completos = []
        for resultado in resultados:
            processo_completo = {
                "nome": nome_map[resultado["nome"]],
                "data_inicio": resultado["data_inicio"].strftime('%d/%m/%Y %H:%M:%S'),
                "data_fim": resultado["data_fim"].strftime('%d/%m/%Y %H:%M:%S'),
                "duracao": format_duracao(resultado["duracao"]),
                "status": resultado["status"]
            }
            processos_completos.append(processo_completo)
        
    except Exception as e:
        print(f"Erro ao obter processos completos: {e}")
        processos_completos = []
    
    # Recuperar o menu e renderizar com dados de paginação
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    # Calcule o número total de páginas
    total_processos_query = "SELECT COUNT(*) FROM processo"
    total_processos_completos_query = "SELECT COUNT(*) FROM processos_completos WHERE EXTRACT(EPOCH FROM duracao) > 10"
    
    total_processos = banco().sql(total_processos_query, ())[0]["count"]
    total_processos_completos = banco().sql(total_processos_completos_query, ())[0]["count"]
    
    total_paginas = (total_processos + processos_por_pagina - 1) // processos_por_pagina
    total_paginas_completos = (total_processos_completos + processos_por_pagina - 1) // processos_por_pagina
    
    
    start_page = max(1, pagina - 5)
    end_page = min(total_paginas_completos, pagina + 5)
    
    return render_template(
        "processos/processos.html",
        usuario=session["usuario"],
        menu=menu,
        processos=processos,
        processos_completos=processos_completos,
        pagina=pagina,
        total_paginas=total_paginas,
        total_paginas_completos=total_paginas_completos,
        start_page=start_page,
        end_page=end_page
    )
