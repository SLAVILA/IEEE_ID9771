from flask import Blueprint

b_home = Blueprint("home", __name__)

from . import home
