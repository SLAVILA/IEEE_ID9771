from flask import Blueprint

b_permissoes = Blueprint("permissoes", __name__)

from . import (
    _permissoes_verifica,
    permissoes_lista,
    permissoes_edicao,
    permissoes_novo,
    _permissoes_altera_une_grupo_permissao,
    _permissoes_altera_todas_perm,
    _permissoes_altera_nome,
    _permissoes_altera_status,
    _permissoes_verifica_exc,
    _permissoes_exclui,
    _permissoes_exclui_select,
    _permissoes_lista,
    _permissoes_novo,
)
