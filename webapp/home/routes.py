# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from webapp.home import blueprint
from flask import render_template
from jinja2 import TemplateNotFound


@blueprint.route('/index')
def index():

    return render_template('index.html')


@blueprint.route('/<template>')
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        return render_template(template)

    except TemplateNotFound:
        return render_template('page-404.html'), 404

    except:
        return render_template('page-500.html'), 500
