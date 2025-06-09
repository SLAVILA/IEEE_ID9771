from flask import render_template, request, redirect, url_for, session
from biblioteca import modulos
import json
from random import randint
import sys
import math


from app.permissoes import b_permissoes

@b_permissoes.route("/permissoes_lista", methods=['GET'])
def permissoes_lista():
    if not 'usuario' in session.keys():
        return redirect('login')
    if not 'nome_cliente' in session['usuario'].keys():
        return redirect("https://"+request.host+"/login")
    menu = modulos.permissao().lista_permissao(session['usuario']['id'])
    if not menu[8][0]:
        session['msg'] = "Você não possui permissão para acessar este módulo..."
        session['msg_type'] = "danger"
        return redirect("https://"+request.host+"/")
    if 'exclui_usuario' in session.keys():
        session.pop('exclui_usuario')
    if 'edita_usuario' in session.keys():
        session.pop('edita_usuario')
    if session['usuario']['admin']:
        resposta = modulos.banco().sql("select g.id,g.str_nome, c.str_nome as nome_cliente, case when g.bol_status is true then \
                                    'ATIVO' else 'INATIVO' end as str_status from grupos g, cliente c where g.fk_id_cliente = c.id",())
        admin=1
    else:
        resposta = modulos.banco().sql("select id,str_nome, case when bol_status is true then \
                                            'ATIVO' else 'INATIVO' end as str_status from grupos where fk_id_cliente=$1" , session['usuario']['id_cliente'])
        admin=0
    if resposta:
        grupos=resposta
    else:
        grupos=[]

    if 'msg' in session.keys():
        msg=session['msg']
        msg_type=session['msg_type']
        session.pop('msg')
        session.pop('msg_type')
    else:
        msg=''
        msg_type=''


    return render_template("permissoes/permissoes_lista.html",usuario=session['usuario'],menu=menu,grupos=grupos,msg=msg,msg_type=msg_type,admin=admin)
