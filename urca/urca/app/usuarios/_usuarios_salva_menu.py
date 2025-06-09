from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_salva_menu", methods=["POST"])
def _usuarios_salva_menu():
    """
    Salva os dados do menu.

    Parâmetros:
    - Nenhum

    Retorna:
    - Nenhum
    """
    # Obtem os dados do formulário
    form = request.form
    menu = form.get("menu")
    
    # if not "menu" in session.keys():
    # O uso de not in aprimora a leitura
    if "menu" not in session.keys():
        session["menu"] = {}

    if menu == "config":
        #if not "config" in session["menu"].keys():
        # O uso de not in aprimora a leitura
        if "config" not in session["menu"].keys():
            session["menu"]["config"] = 1
        else:
            session["menu"].pop("config")
