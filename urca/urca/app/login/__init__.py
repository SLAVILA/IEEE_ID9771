from flask import Blueprint

b_login = Blueprint("login", __name__)

from . import login, _login, login_alteracao_senha, _login_alteracao_senha, logout
