# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_wtf import FlaskForm
from wtforms import TextField, IntegerField, FloatField, BooleanField
from wtforms.validators import DataRequired, NumberRange


class PreferencesForm(FlaskForm):
    debug = BooleanField('debug', id='debug')
    debug_nose = BooleanField('debug nose', id='debug_nose')
    debug_mouth = BooleanField('debug mouth', id='debug_mouth')
    debug_eyes = BooleanField('debug eyes', id='debug_eyes')
    debug_face = BooleanField('debug face', id='debug_face')
    debug_mouse = BooleanField('debug mouse', id='debug_mouse')
    debug_brows = BooleanField('debug brows', id='debug_brows')
    filter = BooleanField('filter', id='filter')
    stickyclick = BooleanField('sticky click', id='stickyclick')

    ebd = FloatField('eyebrow distance', id='ebd')
    smoothness = IntegerField('smoothness', id='smoothness', validators=[NumberRange(1, 8)])
