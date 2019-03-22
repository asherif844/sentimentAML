 
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
    model_path = Model.get_model_path('lencount')
    model = dill.load(open(model_path, 'rb'))
 
def run(raw_data):
    data = raw_data
    y_hat = model(data)
    return str(y_hat)
