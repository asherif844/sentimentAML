 
import json
import os
import pickle

import dill
import numpy as np

from azureml.core.model import Model
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


def init():
    global model
    model_path = Model.get_model_path('ta_model')
    model = dill.load(open(model_path, 'rb'))
 
def run(raw_data):
    data = raw_data
    y_hat = model(data)
    return str(y_hat)

