from flask import Blueprint

b_config_modulos = Blueprint("config_modulos", __name__)

from . import config_modulos_lista,config_modulos_edicao,config_modulos_novo,_config_modulos_edicao,_config_modulos_exclui, \
               _config_modulos_lista,_config_modulos_novo,_config_modulos_verifica,_config_modulos_verifica_exc
