from flask import Blueprint

b_series_temporais = Blueprint("series_temporais", __name__)

from . import \
              _series_temporais_lista, \
              series_temporais_lista, \
              _dados_individual
