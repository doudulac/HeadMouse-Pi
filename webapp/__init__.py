# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import Flask
from importlib import import_module


def register_blueprints(app):
    for module_name in ('base', 'home'):
        module = import_module('webapp.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)


def create_app(config):
    app = Flask(__name__, static_folder='base/static')
    app.config.from_object(config)
    register_blueprints(app)
    return app
