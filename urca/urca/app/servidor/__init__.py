from flask import Blueprint

b_servidor = Blueprint("servidor", __name__)

from . import servidor
