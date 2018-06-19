# -*- coding: utf-8 -*-
from flask import render_template, request, url_for, redirect

import scipy.misc
from project import app
from project.models.model_interface import ModelInterface
import os

PER_PAGE = 20


@app.route('/', methods=['GET', 'POST'])
def index():
    mi = ModelInterface
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        img_file = request.files['image']
        img_file.save('project/data/tmp.png')
        img = scipy.misc.imread('project/data/tmp.png')
        result = mi.predict([img])[0]
        return render_template('index.html', result=result)

# app.jinja_env.globals['url_for_songs'] = url_for_songs
