# -*- coding: utf-8 -*-
"""
Spell Checker
Next Word Prediction
Keyword Density Percentage
Classify Title
Classify Breakig News
Classify Content Type
Predict High Page View Content
Find NER (Named Entity Recognition)
"""

from flask import Flask, render_template, jsonify,request
from flask_cors import CORS
import json
import os
import modules.classifytitle as classifytitle
import modules.predictword as predictword
import modules.verticalclassify as verticalclassify
import modules.classifybreaking as classifybreaking
import modules.classifypopular as classifypopular
from modules.ner import Parser

# JSON display handler.
def display_msg(request_obj, input_obj, field_name):
    post_content = request_obj.args.get(field_name)
    if not request_obj.args.get(field_name):
        post_content = request_obj.form[field_name]
    if not post_content:
        return jsonify({"Error": 'No URL entered'})
    try:
        return jsonify(input_obj(post_content))
    except Exception as e:
        return jsonify({"Error": 'There was an error while processing your request. ' + str(e)})

app = Flask(__name__, '/static')
CORS(app)

# Sample html form to test
@app.route("/test")
def output():
      return render_template("sample_gui.html")

# Classify predict vertical
@app.route('/predict_vertical', methods=['POST'])
def start():
    print(request.headers.get('Content-Type'))
    post_content = request.form['content']    
    if not post_content:
        return jsonify({"Error": 'No data entered'})

    return verticalclassify.classify_vertical(post_content)

# Classify title
@app.route("/classify_title")
def classify_title():
    title = request.args.get('title')
    if not request.args.get('title'):
        title = request.form['title']
    if not title:
        return jsonify({"Error": 'No URL entered'})
    try:
        return jsonify(classifytitle.classify_title(title))
    except Exception as e:
        return jsonify({"Error": 'There was an error while processing your request. ' + str(e)})

# Classify breaking news
@app.route("/classify_breaking")
def classify_breaking():
    return display_msg(request, classifybreaking.classify_title, 'title')

# Classify popular news
@app.route("/classify_popular")
def classify_popular():
    return display_msg(request, classifypopular.classify_title, 'title')

# Classify predict next word
@app.route('/output', methods=['GET'])
def worker():
    string = request.args.get('string')
    work = request.args.get('work')
    return predictword.predict_word(work, string)

# Classify named entity recognition
@app.route("/ner",  methods=['GET', 'POST'])
def ner():
    content = request.args.get('content')
    if not request.args.get('content'):
        content = request.form['content']
    if not content:
        return jsonify({"Error": 'No data entered'})
    try:
        p = Parser()
        p.load_models("models/")
        return jsonify(p.predict(content))
        del p
    except Exception as e:
        return jsonify({"Error": 'There was an error while processing your request. ' + str(e)})

# Web server initiate
if __name__=="__main__":
     app.run(debug=False, host='0.0.0.0', port=5004)
