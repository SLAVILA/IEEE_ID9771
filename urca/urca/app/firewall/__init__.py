from flask import Blueprint

b_firewall = Blueprint("firewall", __name__)

from . import firewall
