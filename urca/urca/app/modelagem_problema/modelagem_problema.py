from flask import render_template, request, session, redirect, current_app
from biblioteca import modulos
import json
import math
from ast import literal_eval
import pandas as pd

from app.modelagem_problema import b_modelagem_problema



@b_modelagem_problema.route("/modelagem_problema", methods=["GET"])
def modelagem_problema():
    # Verifica se o usuário está logado
    # if not 'usuario' in session.keys():
    # O uso de not in aprimora a leitura
    if "usuario" not in session.keys():
        return redirect("login")
    
    menu = modulos.permissao().lista_permissao(session["usuario"]["id"])
    
    return render_template("modelagem_problema/modelagem.html", usuario=session["usuario"], menu=menu)
