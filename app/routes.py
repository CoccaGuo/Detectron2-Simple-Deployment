import os
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import redirect
from flask.helpers import url_for

from engine.interface import predict
from app import app


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/uploader', methods = ['POST'])
def uploader():
    file = request.files['img']
    img = uuid.uuid4().hex + os.path.splitext(file.filename)[1]
    try:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img)
        file.save(img_path)
    except IsADirectoryError as e:
        return "No file selected."
    img, note = predict(app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], img)
    return redirect(url_for('result', img=img, note=note))


@app.route('/result/<img>/<note>')
def result(img, note):
    img_path = url_for('static', filename = 'result/'+img)
    return render_template('result.html', img=img_path, notes=note)


@app.route('/interface', methods=["POST"])
def interface():
    file = request.files.get('img')
    print(request.data)
    if file is None:
        return "File not posted."
    img = uuid.uuid4().hex + os.path.splitext(file.filename)[1]
    try:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img)
        file.save(img_path)
    except IsADirectoryError as e:
        return "No file selected."
    note = predict(app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'], img)
    return {"img": url_for('static', filename = 'result/'+img), "source": note}