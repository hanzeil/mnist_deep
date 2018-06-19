# -*- coding: utf-8 -*-
__version__ = '0.1'
from flask import Flask
from flask_debugtoolbar import DebugToolbarExtension
app = Flask('project')
app.config['SECRET_KEY'] = 'random'
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
app.debug = False
toolbar = DebugToolbarExtension(app)
from project.controllers import *
