from .urca_db import UrcaDb
from .bcb_db import BcbDb, ExpectativasBcb
from .local import Local
from .PesquisaDesenvolvimentoDb import PesquisaDesenvolvimentoDb
from .ampere import AmpereDB


import os
from dotenv import load_dotenv

base_dir = os.path.split(os.path.dirname(__file__))[:-1]
load_dotenv(os.path.join(*base_dir, 'config', '.env'))