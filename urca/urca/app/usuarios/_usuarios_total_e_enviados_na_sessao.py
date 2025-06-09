from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import hashlib
import math

from app.usuarios import b_usuarios


@b_usuarios.route("/_usuarios_total_e_enviados_na_sessao", methods=["POST"])
def _usuarios_total_e_enviados_na_sessao():
    retorno = {
        "tmp_total": session["tmp_total"],
        "tmp_enviados": session["tmp_enviados"],
        "sessao_selecionada": session["sessao_selecionada"]["id"],
    }
    return json.dumps(retorno)
