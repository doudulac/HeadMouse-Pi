# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template, redirect, request, url_for, current_app

from webapp.base import blueprint
from webapp.base.forms import PreferencesForm


@blueprint.route('/')
def route_default():
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/preferences', methods=['GET', 'POST'])
def preferences():
    args = current_app.jinja_env.globals['args']
    form = PreferencesForm(request.form)
    if 'preferences' in request.form:
        for field in iter(form):
            if field.name in args:
                setattr(args, field.name, request.form.get(field.name))
        return render_template('preferences.html', form=form)

    else:
        for field in iter(form):
            if field.name in args:
                field.data = getattr(args, field.name)
        return render_template('preferences.html', form=form)


@blueprint.route('/shutdown')
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


# Errors

@blueprint.errorhandler(404)
def not_found_error(_error):
    return render_template('page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(_error):
    return render_template('page-500.html'), 500
