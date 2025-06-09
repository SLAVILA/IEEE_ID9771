from flask import Blueprint

b_modelagem_problema = Blueprint("modelagem_problema", __name__)

from . import \
              modelagem_problema
