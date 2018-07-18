from Program import ML
from flask import request
from flask import jsonify
from flask import Flask

from keras.models import Sequential
from keras.models import load_model
import keras
import math

import numpy as np


app=Flask(__name__)


@app.route("/predict",methods=['POST'])
def predict():
    var=request.get_json(force=True)
    user=var["id"]
    data=var["data"]

    response={'out':"true"}

    ml=ML(user)
    predicted_values=ml.predict(data)
    #print(predicted_values)
    response={
        '1':str(round(predicted_values[0],2)),
        '2':str(round(predicted_values[1],2)),
        '3':str(round(predicted_values[2],2))
    }
    ml=None
    #print(response)
    return jsonify(response)
    

@app.route('/sample')
def running():
    return 'Flask is Running!'


