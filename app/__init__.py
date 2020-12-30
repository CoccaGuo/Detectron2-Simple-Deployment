from flask import Flask
app = Flask(__name__)
from app import routes

app.config['UPLOAD_FOLDER'] = 'app/static/upload/'
app.config['RESULT_FOLDER'] = 'app/static/result/'