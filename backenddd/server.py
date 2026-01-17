from flask import Flask
import model as m
from flask import Flask, app, render_template, request
from werkzeug.datastructures import FileStorage
from os import remove
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

m.init()


@app.post("/process")
def process() -> dict:
    file: FileStorage = request.files["file"]
    file.save(file.filename)
    print(file.filename)
    x = m.predict_image(file.filename, m.model)
    remove(file.filename)
    return {"cat": m.category[x][1]}
