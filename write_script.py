%%writefile score.py
 
import json
import numpy as np
import os
import pickle
import dill
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
 
from azureml.core.model import Model
 
def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('ta_model')
    model = dill.load(open(model_path, 'rb'))
 
def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = model(data)
    return json.dumps(y_hat.tolist())