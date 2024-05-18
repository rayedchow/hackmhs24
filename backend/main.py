from flask import Flask, request
from flask_cors import CORS, cross_origin
from models.heart_attack import attack_prediction
from models.heart_disease import disease_prediction
from models.heart_disease_stage import stage_prediction
from models.heart_failure import failure_prediction
from models.stroke import stroke_prediction

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/heart_attack", methods=['POST'])
@cross_origin()
def heart_attack():
    data = request.get_json()
    predictedRisk = attack_prediction(data)
    return predictedRisk

@app.route("/heart_disease", methods=['POST'])
@cross_origin()
def heart_disease():
    data = request.get_json()
    predictedRisk = disease_prediction(data)
    return predictedRisk

@app.route("/heart_disease_stage", methods=['POST'])
@cross_origin()
def heart_disease_stage():
    data = request.get_json()
    predictedRisk = stage_prediction(data)
    return predictedRisk

@app.route("/heart_failure", methods=['POST'])
@cross_origin()
def heart_failure():
    data = request.get_json()
    predictedRisk = failure_prediction(data)
    return predictedRisk

@app.route("/stroke", methods=['POST'])
@cross_origin()
def stroke():
    data = request.get_json()
    predictedRisk = stroke_prediction(data)
    return predictedRisk