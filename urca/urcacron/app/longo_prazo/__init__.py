from flask import Blueprint
import os
import importlib

b_longo_prazo = Blueprint("longo_prazo", __name__)

# Identificar o diretório do arquivo atual
current_dir = os.path.dirname(__file__)

# Listar todos os arquivos Python na raiz do diretório atual
python_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']

# Importar cada arquivo Python encontrado
for file in python_files:
    module_name = file[:-3]  # Remove a extensão '.py'
    module = importlib.import_module(f'.{module_name}', package=__name__)
    globals()[module_name] = module